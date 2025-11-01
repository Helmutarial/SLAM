"""
Script to align and visualize two SLAM point clouds and their trajectories in 2D (plan view).
Alignment: detections (SVD) + ICP. Only the final visualization is shown, with subsampled clouds and aligned trajectories.
All code and comments in English.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import json
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.pointcloud_utils.accumulation import merge_point_clouds
from collaborative_slam.utils.pointcloud_utils.alignment import compute_icp_rmse
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

def transform_traj(traj, T):
    """
    Apply a 4x4 transformation matrix to a trajectory (Nx2 or Nx3).
    Returns Nx2 array.
    """
    if traj is None or traj.size == 0:
        return traj
    if traj.shape[1] == 2:
        traj_h = np.hstack([traj, np.zeros((traj.shape[0], 1)), np.ones((traj.shape[0], 1))])
    else:
        traj_h = np.hstack([traj, np.ones((traj.shape[0], 1))])
    traj_t = (T @ traj_h.T).T
    return traj_t[:, :2]

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

    # Load data
    traj1 = load_trajectory(poses_path1)
    traj2 = load_trajectory(poses_path2)
    det1 = load_detections(det_path1, min_conf=0.6)
    det2 = load_detections(det_path2, min_conf=0.6)
    clouds1, _ = load_point_clouds(cloud_dir1)
    clouds2, _ = load_point_clouds(cloud_dir2)
    if not clouds1 or not clouds2:
        print('No clouds found in one of the folders.')
        return
    merged1 = merge_point_clouds(clouds1)
    merged2 = merge_point_clouds(clouds2)


    # Helper for subsampling and centering
    def prepare_clouds(merged1, merged2):
        def filter_points(points, z, max_points=8000):
            if points.shape[0] > max_points:
                idx = np.random.choice(points.shape[0], max_points, replace=False)
                return points[idx], z[idx]
            return points, z
        pts1 = np.asarray(merged1.points)
        pts2 = np.asarray(merged2.points)
        pts1_z = pts1[:, 2]
        pts2_z = pts2[:, 2]
        pts1_vis, pts1_z_vis = filter_points(pts1, pts1_z)
        pts2_vis, pts2_z_vis = filter_points(pts2, pts2_z)
        all_xy = np.vstack([pts1_vis[:, :2], pts2_vis[:, :2]])
        x_min, x_max = np.percentile(all_xy[:, 0], [2, 98])
        y_min, y_max = np.percentile(all_xy[:, 1], [2, 98])
        def mask_range(points):
            return (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        mask1 = mask_range(pts1_vis)
        mask2 = mask_range(pts2_vis)
        pts1_vis, pts1_z_vis = pts1_vis[mask1], pts1_z_vis[mask1]
        pts2_vis, pts2_z_vis = pts2_vis[mask2], pts2_z_vis[mask2]
        return pts1_vis, pts1_z_vis, pts2_vis, pts2_z_vis, x_min, x_max, y_min, y_max

    # 1. Only ICP
    merged2_icp = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with ICP only...')
    rmse_icp, t_icp = compute_icp_rmse(merged2_icp, merged1, np.eye(4))
    merged2_icp.transform(t_icp)
    pts1_vis, pts1_z_vis, pts2_vis, pts2_z_vis, x_min, x_max, y_min, y_max = prepare_clouds(merged1, merged2_icp)
    fig, ax = plt.subplots(figsize=(11, 9))
    sc1 = ax.scatter(pts1_vis[:, 0], pts1_vis[:, 1], s=3, c=pts1_z_vis, cmap='Blues', alpha=0.18, label='Cloud 1')
    sc2 = ax.scatter(pts2_vis[:, 0], pts2_vis[:, 1], s=3, c=pts2_z_vis, cmap='Reds', alpha=0.18, label='Cloud 2 ICP')
    plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label='Z height Cloud 1')
    plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04, label='Z height Cloud 2')
    if traj1 is not None and traj1.size > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=3, label='Trajectory 1')
    if traj2 is not None and traj2.size > 0:
        traj2_icp = transform_traj(traj2, t_icp)
        ax.plot(traj2_icp[:, 0], traj2_icp[:, 1], c='black', lw=3, label='Trajectory 2 ICP')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('ICP only')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

    # 2. Only detections
    merged2_det = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with detections only...')
    t_det = align_by_detections(det1, det2)
    merged2_det.transform(t_det)
    pts1_vis, pts1_z_vis, pts2_vis, pts2_z_vis, x_min, x_max, y_min, y_max = prepare_clouds(merged1, merged2_det)
    fig, ax = plt.subplots(figsize=(11, 9))
    sc1 = ax.scatter(pts1_vis[:, 0], pts1_vis[:, 1], s=3, c=pts1_z_vis, cmap='Blues', alpha=0.18, label='Cloud 1')
    sc2 = ax.scatter(pts2_vis[:, 0], pts2_vis[:, 1], s=3, c=pts2_z_vis, cmap='Reds', alpha=0.18, label='Cloud 2 detections')
    plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label='Z height Cloud 1')
    plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04, label='Z height Cloud 2')
    if traj1 is not None and traj1.size > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=3, label='Trajectory 1')
    if traj2 is not None and traj2.size > 0:
        traj2_det = transform_traj(traj2, t_det)
        ax.plot(traj2_det[:, 0], traj2_det[:, 1], c='black', lw=3, label='Trajectory 2 detections')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('Detections only')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

    # 3. Combined (detections + ICP)
    merged2_comb = merge_point_clouds(clouds2) # fresh copy
    print('Aligning with detections + ICP...')
    t_total = align_by_detections(det1, det2)
    merged2_comb.transform(t_total)
    rmse_comb, t_icp_comb = compute_icp_rmse(merged2_comb, merged1, np.eye(4))
    t_total = t_icp_comb @ t_total
    merged2_comb.transform(t_icp_comb)
    pts1_vis, pts1_z_vis, pts2_vis, pts2_z_vis, x_min, x_max, y_min, y_max = prepare_clouds(merged1, merged2_comb)
    fig, ax = plt.subplots(figsize=(11, 9))
    sc1 = ax.scatter(pts1_vis[:, 0], pts1_vis[:, 1], s=3, c=pts1_z_vis, cmap='Blues', alpha=0.18, label='Cloud 1')
    sc2 = ax.scatter(pts2_vis[:, 0], pts2_vis[:, 1], s=3, c=pts2_z_vis, cmap='Reds', alpha=0.18, label='Cloud 2 det+ICP')
    plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label='Z height Cloud 1')
    plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04, label='Z height Cloud 2')
    if traj1 is not None and traj1.size > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=3, label='Trajectory 1')
    if traj2 is not None and traj2.size > 0:
        traj2_comb = transform_traj(traj2, t_total)
        ax.plot(traj2_comb[:, 0], traj2_comb[:, 1], c='black', lw=3, label='Trajectory 2 det+ICP')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('Detections + ICP')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
