"""
Script to associate 2D detections with 3D point clouds and project detections to 3D coordinates.
- Loads detections from detections.json
- Loads point clouds and camera poses
- Uses calibration data to project detections to 3D
- Saves results in detections_3d.json
"""
# Imports
import os
import json
import numpy as np
from collaborative_slam.utils.file_utils import select_data_folder
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.trayectory_utils.trajectory_filtering import load_camera_trajectory
from collaborative_slam.tools.association_methods.associate_detections_to_depthmap import associate_detections_with_depth

def load_calibration(calib_path):
    """
    Loads camera calibration matrix from JSON file.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    # Usar la primera cámara principal
    cam = calib['cameras'][0]
    camera_matrix = np.array([
        [cam['focalLengthX'], 0, cam['principalPointX']],
        [0, cam['focalLengthY'], cam['principalPointY']],
        [0, 0, 1]
    ])
    return camera_matrix

def project_detection_to_3d(centroid_2d, cloud, camera_matrix):
    """
    Projects a 2D detection centroid to 3D using the point cloud and camera matrix.
    Returns the closest 3D point in the cloud to the reprojected ray.
    """
    # Simple approach: find the 3D point whose projection is closest to centroid_2d
    # (Assumes cloud is Nx3 numpy array)
    if cloud.shape[0] == 0:
        return None
    # Project all 3D points to 2D
    points_2d = (camera_matrix @ cloud.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    dists = np.linalg.norm(points_2d - centroid_2d, axis=1)
    idx = np.argmin(dists)
    return cloud[idx].tolist()

def main():
    print("Selecciona la carpeta de entrada...")
    results_folder = select_data_folder()
    detections_path = os.path.join(results_folder, 'detections', 'frames_detected', 'detections.json')
    cloud_dir = os.path.join(results_folder, 'cloud_points')
    calib_path = os.path.join(results_folder, 'calibration.json')
    poses_path = os.path.join(results_folder, 'poses.json')
    output_path = os.path.join(results_folder, 'detections_3d.json')
    output_path_depth = os.path.join(results_folder, 'detections_3d_depthmap.json')
    depthmap_dir = os.path.join(results_folder, 'depth_maps')
    # Load detections
    with open(detections_path, 'r') as f:
        detections = json.load(f)
    # Load clouds
    clouds, files = load_point_clouds(cloud_dir)
    # Load calibration
    camera_matrix = load_calibration(calib_path)

    # Asociar detecciones usando depth map
    detections_3d_depth = associate_detections_with_depth(detections, depthmap_dir, camera_matrix)
    # Load poses (not used in this simple version)
    trajectory = load_camera_trajectory(poses_path)
    # Associate detections to clouds
    detections_3d = []
    for det in detections:
        frame_idx = det.get('frame')
        if 'centroid' not in det or frame_idx is None:
            continue
        # Find cloud for this frame
        cloud_idx = None
        for i, fname in enumerate(files):
            if str(frame_idx) in fname:
                cloud_idx = i
                break
        if cloud_idx is None:
            continue
        cloud = np.asarray(clouds[cloud_idx].points)
        centroid_2d = np.array(det['centroid'])
        point_3d = project_detection_to_3d(centroid_2d, cloud, camera_matrix)
        if point_3d is not None:
            det_3d = det.copy()
            det_3d['point_3d'] = point_3d
            detections_3d.append(det_3d)
    # Guardar ambos resultados
    with open(output_path, 'w') as f:
        json.dump(detections_3d, f, indent=2)
    print(f"Detecciones 3D (cloud) guardadas en {output_path}")
    with open(output_path_depth, 'w') as f:
        json.dump(detections_3d_depth, f, indent=2)
    print(f"Detecciones 3D (depth map) guardadas en {output_path_depth}")

if __name__ == "__main__":
    main()
