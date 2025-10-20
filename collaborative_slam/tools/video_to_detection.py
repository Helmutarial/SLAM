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
        def load_intrinsics(calib_path):
            with open(calib_path, 'r') as f:
                calib = json.load(f)
            cam = calib['cameras'][0]
            K = np.array([
                [cam['focalLengthX'], 0, cam['principalPointX']],
                [0, cam['focalLengthY'], cam['principalPointY']],
                [0, 0, 1]
            ])
            return K
        def pixel_to_ray(K, pixel):
            x, y = pixel
            invK = np.linalg.inv(K)
            pix_h = np.array([x, y, 1.0])
            ray = invK @ pix_h
            ray = ray / np.linalg.norm(ray)
            return ray
        matched = match_detections_to_poses(poses, detections)
        K = load_intrinsics(calib_path)
        results = []
        for det in matched:
            if 'pose_index' in det:
                del det['pose_index']
            pose = det['pose']
            if pose is None:
                det['pose'] = None
                det['closest_point'] = None
                results.append(det)
                continue
            frame_idx = pose.get('frame', 0)
            det['pose'] = pose
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
                cam_pos = np.array([pose['x'], pose['y'], pose['z']])
                vecs = points - cam_pos
                cross = np.cross(vecs, ray)
                dists_to_ray = np.linalg.norm(cross, axis=1) / np.linalg.norm(ray)
                idx = np.argmin(dists_to_ray)
                min_dist = dists_to_ray[idx]
                print(f"Frame {frame_idx} | Detección en ({cx},{cy}) | Distancia mínima al rayo: {min_dist:.3f}")
                closest_pt = points[idx].tolist()
                det['closest_point'] = closest_pt
            else:
                det['closest_point'] = None
            results.append(det)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"detections_3d.json generado correctamente con {len(results)} elementos.")
    else:
        print('No se encontró el archivo de detecciones generado.')

if __name__ == "__main__":
    main()
