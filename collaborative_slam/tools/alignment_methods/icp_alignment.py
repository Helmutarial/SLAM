"""
Script to align and visualize two SLAM point clouds and their trajectories in 2D (plan view).
Alignment: detections (SVD) + ICP. Only the final visualization is shown, with subsampled clouds and aligned trajectories.
All code and comments in English.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collaborative_slam.views.planview_visualization import visualize_planview
import open3d as o3d
import json
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.pointcloud_utils.accumulation import merge_point_clouds
from collaborative_slam.utils.pointcloud_utils.alignment import compute_icp_rmse, transform_trajectory
from collaborative_slam.utils.file_utils import select_data_folder

def load_trajectory(poses_path):
    """
    Load camera trajectory from poses.json as Nx2 array (X, Y).
    """
    if not os.path.exists(poses_path):
        return None
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    return np.array([[p['x'], p['y']] for p in poses])

def load_detections(detections_path, min_conf=0.6):
    """
    Load 3D detections from detections_3d.json, filter by confidence.
    Returns: dict[class] = list of (x, y, z)
    """
    if not os.path.exists(detections_path):
        return {}
    with open(detections_path, 'r') as f:
        dets = json.load(f)
    det_by_class = {}
    for d in dets:
        if d.get('confidence', 0) >= min_conf and 'class' in d and 'point_3d' in d and d['point_3d']:
            det_by_class.setdefault(d['class'], []).append(np.array(d['point_3d']))
    return det_by_class

def load_all_keypoints_from_dir(keypoints_dir):
    """
    Load all keypoints from .npz files in a directory and return as a single (N,3) numpy array.
    """
    import glob
    import numpy as np
    all_points = []
    npz_files = sorted(glob.glob(os.path.join(keypoints_dir, 'keypoints_*.npz')))
    for f in npz_files:
        npz = np.load(f)
        if 'keypoints_camera' in npz:
            all_points.append(npz['keypoints_camera'])
    if not all_points:
        return np.zeros((0, 3))
    return np.vstack(all_points)

def align_by_detections(det1, det2):
    """
    Align two sets of detections (dict[class] = list of points) using SVD (Procrustes).
    Returns: transformation matrix (4x4)
    """
    matches1, matches2 = [], []
    for cls in det1:
        if cls in det2:
            n = min(len(det1[cls]), len(det2[cls]))
            for i in range(n):
                matches1.append(det1[cls][i][:3])
                matches2.append(det2[cls][i][:3])
    if len(matches1) < 3:
        print('Not enough common detections to align.')
        return np.eye(4)
    A = np.stack(matches1)
    B = np.stack(matches2)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = BB.T @ AA
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = centroid_A - R @ centroid_B
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def main():
    print("Select the first folder (video 1)...")
    folder1 = select_data_folder()
    print("Select the second folder (video 2)...")
    folder2 = select_data_folder()
    poses_path1 = os.path.join(folder1, 'poses.json')
    poses_path2 = os.path.join(folder2, 'poses.json')
    det_path1 = os.path.join(folder1, 'detections_3d.json')
    det_path2 = os.path.join(folder2, 'detections_3d.json')
    cloud_dir1 = os.path.join(folder1, 'cloud_points')
    cloud_dir2 = os.path.join(folder2, 'cloud_points')
    keypoints_dir1 = os.path.join(folder1, 'keypoints')
    keypoints_dir2 = os.path.join(folder2, 'keypoints')
    
    # Load data
    traj1 = load_trajectory(poses_path1)
    traj2 = load_trajectory(poses_path2)
    det1 = load_detections(det_path1, min_conf=0.6)
    det2 = load_detections(det_path2, min_conf=0.6)
    clouds1, _ = load_point_clouds(cloud_dir1)
    clouds2, _ = load_point_clouds(cloud_dir2)
    merged1 = merge_point_clouds(clouds1)
    merged2 = merge_point_clouds(clouds2)
    kps1 = load_all_keypoints_from_dir(keypoints_dir1)
    kps2 = load_all_keypoints_from_dir(keypoints_dir2)
    kpcloud1 = o3d.geometry.PointCloud()
    kpcloud2 = o3d.geometry.PointCloud()
    kpcloud1.points = o3d.utility.Vector3dVector(kps1)
    kpcloud2.points = o3d.utility.Vector3dVector(kps2)
    
    # 1. Only ICP
    merged2_icp = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with ICP only...')
    rmse_icp, t_icp = compute_icp_rmse(merged2_icp, merged1, np.eye(4))
    merged2_icp.transform(t_icp)
    pts1 = np.asarray(merged1.points)
    pts2 = np.asarray(merged2_icp.points)
    traj2_icp = transform_trajectory(traj2, t_icp) if traj2 is not None and traj2.size > 0 else None
    visualize_planview(
        pts1, pts2, traj1, traj2_icp,
        label1='Cloud 1', label2='Cloud 2 ICP',
        traj_color1='orange', traj_color2='black',
        title='ICP only'
    )

    # 2. Only detections
    merged2_det = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with detections only...')
    t_det = align_by_detections(det1, det2)
    merged2_det.transform(t_det)
    pts1 = np.asarray(merged1.points)
    pts2 = np.asarray(merged2_det.points)
    traj2_det = transform_trajectory(traj2, t_det) if traj2 is not None and traj2.size > 0 else None
    visualize_planview(
        pts1, pts2, traj1, traj2_det,
        label1='Cloud 1', label2='Cloud 2 detections',
        traj_color1='orange', traj_color2='black',
        title='Detections only'
    )

    # 3. Combined (detections + ICP)
    merged2_comb = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with detections + ICP...')
    t_total = align_by_detections(det1, det2)
    merged2_comb.transform(t_total)
    rmse_comb, t_icp_comb = compute_icp_rmse(merged2_comb, merged1, np.eye(4))
    t_total = t_icp_comb @ t_total
    merged2_comb.transform(t_icp_comb)
    pts1 = np.asarray(merged1.points)
    pts2 = np.asarray(merged2_comb.points)
    traj2_comb = transform_trajectory(traj2, t_total) if traj2 is not None and traj2.size > 0 else None
    visualize_planview(
        pts1, pts2, traj1, traj2_comb,
        label1='Cloud 1', label2='Cloud 2 det+ICP',
        traj_color1='orange', traj_color2='black',
        title='Detections + ICP'
    )

    # Paso 4: alineamiento ICP de los keypoints y visualización
    if len(kpcloud1.points) > 10 and len(kpcloud2.points) > 10:
        print('Aligning keypoints clouds with ICP...')
        kpcloud2_icp = o3d.geometry.PointCloud(kpcloud2)
        rmse_kp, t_kp = compute_icp_rmse(kpcloud2_icp, kpcloud1, np.eye(4))
        kpcloud2_icp.transform(t_kp)
        pts1 = np.asarray(kpcloud1.points)
        pts2 = np.asarray(kpcloud2_icp.points)
        traj2_kp = transform_trajectory(traj2, t_kp) if traj2 is not None and traj2.size > 0 else None
        visualize_planview(
            pts1, pts2, traj1, traj2_kp,
            label1='Keypoints 1', label2='Keypoints 2 ICP',
            traj_color1='orange', traj_color2='blue',
            title='ICP with USIP keypoints (global)'
        )
    else:
        print('No keypoints found or too few for ICP alignment.')


    # Paso 5: alinear keypoints usando la transformación de detecciones (como en el paso 2)
    if len(kpcloud1.points) > 10 and len(kpcloud2.points) > 10:
        print('Aligning keypoints clouds with detections only...')
        t_det_kp = align_by_detections(det1, det2)
        kpcloud2_det = o3d.geometry.PointCloud(kpcloud2)
        kpcloud2_det.transform(t_det_kp)
        pts1 = np.asarray(kpcloud1.points)
        pts2 = np.asarray(kpcloud2_det.points)
        traj2_kp_det = transform_trajectory(traj2, t_det_kp) if traj2 is not None and traj2.size > 0 else None
        visualize_planview(
            pts1, pts2, traj1, traj2_kp_det,
            label1='Keypoints 1', label2='Keypoints 2 detections',
            traj_color1='orange', traj_color2='blue',
            title='Keypoints: detections only'
        )
    else:
        print('No keypoints found or too few for detections alignment.')

    # Paso 6: alinear keypoints primero con detecciones y luego con ICP (combinado)
    if len(kpcloud1.points) > 10 and len(kpcloud2.points) > 10:
        print('Aligning keypoints clouds with detections + ICP...')
        t_det_kp = align_by_detections(det1, det2)
        kpcloud2_comb = o3d.geometry.PointCloud(kpcloud2)
        kpcloud2_comb.transform(t_det_kp)
        rmse_comb_kp, t_icp_comb_kp = compute_icp_rmse(kpcloud2_comb, kpcloud1, np.eye(4))
        t_total_kp = t_icp_comb_kp @ t_det_kp
        kpcloud2_comb.transform(t_icp_comb_kp)
        pts1 = np.asarray(kpcloud1.points)
        pts2 = np.asarray(kpcloud2_comb.points)
        traj2_kp_comb = transform_trajectory(traj2, t_total_kp) if traj2 is not None and traj2.size > 0 else None
        visualize_planview(
            pts1, pts2, traj1, traj2_kp_comb,
            label1='Keypoints 1', label2='Keypoints 2 det+ICP',
            traj_color1='orange', traj_color2='blue',
            title='Keypoints: detections + ICP'
        )
    else:
        print('No keypoints found or too few for detections+ICP alignment.')
        
if __name__ == "__main__":
    main()
