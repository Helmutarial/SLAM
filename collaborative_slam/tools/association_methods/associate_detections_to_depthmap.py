"""
Script to associate 2D detections with depth maps and reconstruct 3D positions using camera calibration.
- Loads detections from detections.json
- Loads depth maps and calibration
- For each detection, assigns depth from depth map and reconstructs 3D position
- Returns list of detections with 3D positions from depth map
"""
# Imports
import os
import json
import numpy as np
import cv2

def load_calibration(calib_path):
    """
    Loads camera calibration matrix from JSON file.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    cam = calib['cameras'][0]
    camera_matrix = np.array([
        [cam['focalLengthX'], 0, cam['principalPointX']],
        [0, cam['focalLengthY'], cam['principalPointY']],
        [0, 0, 1]
    ])
    return camera_matrix

def reconstruct_3d_from_depth(xy, depth, camera_matrix):
    """
    Reconstructs 3D position from 2D pixel and depth using camera intrinsics.
    """
    x, y = xy
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    Z = depth
    if Z <= 0:
        return None
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return [X, Y, Z]

def associate_detections_with_depth(detections, depthmap_dir, camera_matrix):
    """
    For each detection, assigns depth from depth map and reconstructs 3D position.
    Returns list of detections with 3D positions from depth map.
    """
    detections_3d_depth = []
    total = 0
    procesadas = 0
    descartadas = 0
    for det in detections:
        frame_idx = det.get('frame')
        if 'centroid' not in det or frame_idx is None:
            descartadas += 1
            continue
        total += 1
        depth_path = os.path.join(depthmap_dir, f"{frame_idx}_depth.png")
        if not os.path.exists(depth_path):
            descartadas += 1
            continue
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        x, y = map(int, det['centroid'])
        if not (0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]):
            descartadas += 1
            continue
        depth = float(depth_map[y, x])
        # Si la imagen está normalizada (0-255), desnormalizar a rango real
        if depth_map.dtype == np.uint8:
            # Recuperar rango real de profundidad (ejemplo: usar min/max del mapa)
            min_val = np.min(depth_map[depth_map > 0]) if np.count_nonzero(depth_map) > 0 else 0
            max_val = np.max(depth_map)
            if max_val > min_val:
                depth = min_val + (max_val - min_val) * (depth / 255.0)
        if depth <= 0:
            descartadas += 1
            continue
        point_3d = reconstruct_3d_from_depth((x, y), depth, camera_matrix)
        if point_3d is not None:
            det_3d = det.copy()
            det_3d['point_3d_depthmap'] = point_3d
            detections_3d_depth.append(det_3d)
            procesadas += 1
        else:
            descartadas += 1
    print(f"Total detecciones: {total}, procesadas: {procesadas}, descartadas: {descartadas}")
    return detections_3d_depth
