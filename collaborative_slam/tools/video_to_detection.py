"""
Script para unificar el proceso de detección de objetos en vídeo usando YOLO.
- Solicita al usuario la carpeta de entrada.
- Busca el vídeo 'rgb_video.mp4' en la carpeta seleccionada.
- Extrae frames y ejecuta YOLO sobre ellos.
- Guarda el JSON de detecciones en results/<nombre_carpeta>/detections_3d.json
"""
# Imports

import os
import sys
import json
import numpy as np
import open3d as o3d
from collaborative_slam.utils.file_utils import select_data_folder
from collaborative_slam.utils.video_utils.extract_frames import extract_frames
from collaborative_slam.utils.yolo_utils.detect_objects import detect_objects_in_folder
from collaborative_slam.utils.frame_matching_utils import match_detections_to_poses

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')


def main():
    # Select input folder
    print("Selecciona la carpeta de entrada de datos...")
    input_folder = select_data_folder()
    if not input_folder:
        print("No se seleccionó ninguna carpeta.")
        sys.exit(1)
    video_path = os.path.join(input_folder, "rgb_video.mp4")
    if not os.path.exists(video_path):
        print(f"No se encontró el vídeo rgb_video.mp4 en {input_folder}")
        sys.exit(1)
    folder_name = os.path.basename(os.path.normpath(input_folder))
    out_dir = os.path.join(RESULTS_ROOT, folder_name, 'detections')
    frames_dir = os.path.join(out_dir, 'frames')
    detected_dir = os.path.join(out_dir, 'frames_detected')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(detected_dir, exist_ok=True)
    print(f'Extrayendo frames de {video_path}...')
    extract_frames(video_path, frames_dir, every_n=30, ext='jpg')
    print('Detectando objetos en los frames...')
    detect_objects_in_folder(frames_dir, detected_dir, model_path=YOLO_MODEL_PATH)
    # El JSON de detecciones se guarda solo en frames_detected/detections.json
    detections_json_path = os.path.join(detected_dir, 'detections.json')
    if os.path.exists(detections_json_path):
        print(f'Detecciones guardadas en {detections_json_path}')
        # Integrar asociación detección-nube de puntos y proyección 3D
        poses_json_path = os.path.join(RESULTS_ROOT, folder_name, 'poses.json')
        cloud_dir = os.path.join(RESULTS_ROOT, folder_name, 'cloud_points')
        calib_path = os.path.join(input_folder, 'calibration.json')
        output_json = os.path.join(RESULTS_ROOT, folder_name, 'detections_3d.json')
        if not os.path.exists(poses_json_path):
            print(f'No se encontró poses.json en {poses_json_path}. No se puede asociar detecciones a nubes.')
            return
        if not os.path.exists(calib_path):
            print(f'No se encontró calibration.json en {calib_path}. No se puede proyectar a 3D.')
            return
        with open(detections_json_path, 'r') as f:
            detections = json.load(f)
        with open(poses_json_path, 'r') as f:
            poses = json.load(f)
        def load_intrinsics_and_imu(calib_path):
            """
            Load camera intrinsics and IMU-to-camera transformation matrix.
            """
            with open(calib_path, 'r') as f:
                calib = json.load(f)
            cam = calib['cameras'][0]
            K = np.array([
                [cam['focalLengthX'], 0, cam['principalPointX']],
                [0, cam['focalLengthY'], cam['principalPointY']],
                [0, 0, 1]
            ])
            imu_to_cam = np.array(cam['imuToCamera'])[:3, :3]
            return K, imu_to_cam

        def pixel_to_ray(K, pixel):
            """
            Project pixel coordinates to normalized camera ray.
            """
            x, y = pixel
            invK = np.linalg.inv(K)
            pix_h = np.array([x, y, 1.0])
            ray = invK @ pix_h
            ray = ray / np.linalg.norm(ray)
            return ray

        def load_gyro_orientations(data_jsonl_path):
            """
            Load gyroscope orientations for each frame from data.jsonl.
            Returns a dict: {frame_idx: gyro_vector}
            """
            import json
            gyro_by_time = {}
            with open(data_jsonl_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'sensor' in entry and entry['sensor']['type'] == 'gyroscope':
                            time = entry['time']
                            gyro = entry['sensor']['values']
                            gyro_by_time[time] = gyro
                    except Exception:
                        continue
            return gyro_by_time

        def get_closest_gyro(frame_time, gyro_by_time):
            """
            Find the closest gyro orientation for a given frame time.
            """
            times = np.array(list(gyro_by_time.keys()))
            idx = np.argmin(np.abs(times - frame_time))
            return gyro_by_time[times[idx]]

        def rotate_ray_with_gyro(ray, gyro, imu_to_cam):
            """
            Rotate the camera ray using gyro orientation and imu_to_cam matrix.
            """
            # For simplicity, treat gyro as a direction vector (approximation)
            # In a real scenario, convert gyro to rotation matrix or quaternion
            # Here, we use the gyro vector as the direction and rotate with imu_to_cam
            direction = np.array(gyro)
            direction = direction / np.linalg.norm(direction)
            rotated_ray = imu_to_cam @ direction
            rotated_ray = rotated_ray / np.linalg.norm(rotated_ray)
            return rotated_ray
        matched = match_detections_to_poses(poses, detections)
        K, imu_to_cam = load_intrinsics_and_imu(calib_path)
        data_jsonl_path = os.path.join(input_folder, 'data.jsonl')
        gyro_by_time = load_gyro_orientations(data_jsonl_path) if os.path.exists(data_jsonl_path) else {}
        results = []
        for det in matched:
            if 'pose_index' in det:
                del det['pose_index']
            pose = det['pose']
            if pose is None:
                det['pose'] = None
                det['closest_point'] = None
                det['pose_frame'] = None
                results.append(det)
                continue
            frame_idx = pose.get('frame', 0)
            det['pose'] = pose
            det['pose_frame'] = frame_idx
            cloud_path = os.path.join(cloud_dir, f'{frame_idx}.ply')
            if not os.path.exists(cloud_path):
                det['closest_point'] = None
                results.append(det)
                continue
            cloud = o3d.io.read_point_cloud(cloud_path)
            points = np.asarray(cloud.points)
            if 'centroid' in det:
                cx, cy = det['centroid']
                ray = pixel_to_ray(K, (cx, cy))
                # Get gyro orientation for this frame (approximate by frame_idx)
                frame_time = None
                # Try to get time from pose if available
                if 'time' in pose:
                    frame_time = pose['time']
                # If not, use frame_idx as proxy (not exact)
                if frame_time is None:
                    frame_time = float(frame_idx)
                gyro = get_closest_gyro(frame_time, gyro_by_time) if gyro_by_time else np.array([0,0,1])
                rotated_ray = rotate_ray_with_gyro(ray, gyro, imu_to_cam)
                cam_pos = np.array([pose['x'], pose['y'], pose['z']])
                vecs = points - cam_pos
                cross = np.cross(vecs, rotated_ray)
                dists_to_ray = np.linalg.norm(cross, axis=1) / np.linalg.norm(rotated_ray)
                idx = np.argmin(dists_to_ray)
                min_dist = dists_to_ray[idx]
                print(f"Frame {frame_idx} | Detection at ({cx},{cy}) | Min distance to ray: {min_dist:.3f}")
                closest_pt = points[idx].tolist()
                det['closest_point'] = closest_pt
            else:
                det['closest_point'] = None
            results.append(det)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"detections_3d.json generated with {len(results)} elements.")
    else:
        print('No se encontró el archivo de detecciones generado.')

if __name__ == "__main__":
    main()
