"""
Script para asociar detecciones con nubes de puntos y proyectarlas a 3D.
Lee detections.json y poses.json, realiza el matching y la proyección, y guarda detections_3d.json.
"""

# Imports
import os
import json
import numpy as np
import open3d as o3d
from collaborative_slam.utils.frame_matching_utils import match_detections_to_poses

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

if __name__ == "__main__":
    # Configuración de rutas
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1')
    detections_json = os.path.join(base_dir, 'detections', 'frames_detected', 'detections.json')
    poses_json = os.path.join(base_dir, 'poses.json')
    cloud_dir = os.path.join(base_dir, 'cloud_points')
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'VIDEO1', 'calibration.json')
    output_json = os.path.join(base_dir, 'detections_3d.json')

    # Cargar detecciones y poses
    with open(detections_json, 'r') as f:
        detections = json.load(f)
    with open(poses_json, 'r') as f:
        poses = json.load(f)

    # Matching detección-pose
    matched = match_detections_to_poses(poses, detections)
    K = load_intrinsics(calib_path)
    results = []
    for det in matched:
        # Eliminar la clave 'pose_index' si existe
        if 'pose_index' in det:
            del det['pose_index']
        pose = det['pose']
        # Guardar la pose completa (diccionario con x, y, z y frame)
        if pose is None:
            det['pose'] = None
            det['closest_point'] = None
            results.append(det)
            continue
        frame_idx = pose.get('frame', 0)
        det['pose'] = pose  # Guardar el diccionario completo
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
            # Relajar el criterio: guardar el punto más cercano siempre, y opcionalmente marcar si la distancia es alta
            closest_pt = points[idx].tolist()
            det['closest_point'] = closest_pt
        else:
            det['closest_point'] = None
        results.append(det)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"detections_3d.json generado correctamente con {len(results)} elementos.")
