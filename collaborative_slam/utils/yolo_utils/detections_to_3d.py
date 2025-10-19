"""
Proyecta las detecciones YOLO (centroides en píxeles) a coordenadas 3D usando la calibración y la pose de cada frame.
Asocia cada detección al punto de la nube más cercano.
"""
import os
import json
import numpy as np
import open3d as o3d

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

def project_detections_to_3d(detections_folder, poses_path, calib_path, cloud_folder, output_json):
    K = load_intrinsics(calib_path)
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    results = []
    for i, pose in enumerate(poses):
        frame_idx = pose.get('frame', i)
        det_path = os.path.join(detections_folder, f'frame_{frame_idx:05d}.json')
        cloud_path = os.path.join(cloud_folder, f'{frame_idx}.ply')
        if not os.path.exists(det_path) or not os.path.exists(cloud_path):
            continue
        with open(det_path, 'r') as f:
            detections = json.load(f)
        cloud = o3d.io.read_point_cloud(cloud_path)
        points = np.asarray(cloud.points)
        for det in detections:
            cx, cy = det['centroid']
            ray = pixel_to_ray(K, (cx, cy))
            cam_pos = np.array([pose['x'], pose['y'], pose['z']])
            # Buscar el punto de la nube más cercano al rayo proyectado desde la cámara
            # Para cada punto, calcular la distancia al rayo: ||(p - cam_pos) x ray|| / ||ray||
            vecs = points - cam_pos
            cross = np.cross(vecs, ray)
            dists_to_ray = np.linalg.norm(cross, axis=1) / np.linalg.norm(ray)
            idx = np.argmin(dists_to_ray)
            closest_pt = points[idx].tolist()
            det3d = det.copy()
            det3d['frame'] = frame_idx
            det3d['camera_position'] = cam_pos.tolist()
            det3d['closest_point'] = closest_pt
            results.append(det3d)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Detecciones proyectadas a 3D guardadas en {output_json}')
