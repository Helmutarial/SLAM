import sys
from collaborative_slam.utils.file_utils import select_data_folder
"""
Real-time 2D wall visualization from point cloud data.

This script processes point clouds from OAK-D recordings and generates a top-down (plan view) map, detecting and drawing walls as they are observed by the camera.

Main steps:
1. Load or receive point clouds in real time.
2. Project points to the XY plane (ignore Z).
3. Detect linear structures (walls) using clustering and line fitting (e.g., RANSAC).
4. Update and visualize the map in 2D as new data arrives.

Requirements: numpy, matplotlib, open3d (for point cloud handling)
"""

# Imports

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
from sklearn.linear_model import RANSACRegressor

def project_to_xy(point_cloud):
    """
    Projects a 3D point cloud to the XY plane.
    Args:
        point_cloud (np.ndarray): Nx3 array of 3D points.
    Returns:
        np.ndarray: Nx2 array of 2D points (X, Y).
    """
    return point_cloud[:, :2]


def detect_walls(points_2d, min_samples=50, residual_threshold=0.05):
    """
    Detects linear structures (walls) in 2D points using RANSAC.
    Args:
        points_2d (np.ndarray): Nx2 array of 2D points.
    Returns:
        list: List of wall segments as (slope, intercept, inlier_points).
    """
    walls = []
    if len(points_2d) < min_samples:
        return walls
    X = points_2d[:,0].reshape(-1,1)
    y = points_2d[:,1]
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
    try:
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        inlier_points = points_2d[inlier_mask]
        walls.append((slope, intercept, inlier_points))
    except Exception as e:
        print(f"RANSAC failed: {e}")
    return walls


def visualize_map(points_2d, walls):
    """
    Plots the 2D map with detected walls.
    Args:
        points_2d (np.ndarray): Nx2 array of 2D points.
        walls (list): List of wall segments.
    """
    plt.figure(figsize=(8,8))
    plt.scatter(points_2d[:,0], points_2d[:,1], s=1, c='gray', alpha=0.5, label='Points')
    for slope, intercept, inlier_points in walls:
        x_vals = np.linspace(np.min(points_2d[:,0]), np.max(points_2d[:,0]), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Detected wall')
        plt.scatter(inlier_points[:,0], inlier_points[:,1], s=5, c='blue', alpha=0.7, label='Wall inliers')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Real-time Wall Mapping')
    plt.legend()
    plt.grid(True)
    plt.show()


# ...existing code...

def main():
    """
    Main function to process point clouds and visualize walls in real time.
    """
    print("Select the folder containing the point clouds (.ply files) to visualize walls.")
    data_folder = select_data_folder()
    ply_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.ply')])
    if not ply_files:
        print("No .ply files found in the selected folder.")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Real-time Wall Mapping (Frame 1/{})'.format(len(ply_files)))
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-25, 7)
    ax.set_ylim(-20, 8)

    all_detected_walls = []
    all_inliers = []

    import matplotlib.gridspec as gridspec
    # Crear figura con dos subplots: 3D (izquierda) y 2D (derecha)
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

    for idx, ply_file in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(os.path.join(data_folder, ply_file))
        points = np.asarray(pcd.points)
        # Calcular el suelo como el percentil bajo de Z
        z_floor = np.percentile(points[:,2], 10)
        # Umbral para considerar pared (ajustable)
        wall_threshold = z_floor + 0.25
        # Separar suelo y posibles paredes
        floor_mask = points[:,2] <= wall_threshold
        wall_mask = points[:,2] > wall_threshold
        floor_points = points[floor_mask]
        wall_points = points[wall_mask]
        points_2d = project_to_xy(points)
        floor_points_2d = project_to_xy(floor_points)
        wall_points_2d = project_to_xy(wall_points)
        # Detectar paredes solo en puntos elevados
        walls = detect_walls(wall_points_2d)

        # --- Visualización 3D ---
        ax3d.clear()
        ax3d.set_title(f'3D Pointcloud (Frame {idx+1}/{len(ply_files)})')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.grid(True)
        # Pintar suelo en gris y paredes en color
        ax3d.scatter(floor_points[:, 0], floor_points[:, 1], floor_points[:, 2], s=1, c='gray', alpha=0.3)
        if wall_points.shape[0] > 0:
            ax3d.scatter(wall_points[:, 0], wall_points[:, 1], wall_points[:, 2], s=8, c='red', alpha=0.7)
        ax3d.set_xlim(-25, 7)
        ax3d.set_ylim(-20, 8)
        ax3d.set_zlim(np.min(points[:,2]), np.max(points[:,2]))

        # --- Visualización 2D ---
        ax2d.clear()
        ax2d.set_title(f'Wall Mapping 2D (Frame {idx+1}/{len(ply_files)})')
        ax2d.set_xlabel('X (meters)')
        ax2d.set_ylabel('Y (meters)')
        ax2d.grid(True)
        ax2d.set_aspect('equal')
        ax2d.set_xlim(-25, 7)
        ax2d.set_ylim(-20, 8)
        ax2d.scatter(floor_points_2d[:, 0], floor_points_2d[:, 1], s=1, c='gray', alpha=0.3, label='Floor points')
        if wall_points_2d.shape[0] > 0:
            ax2d.scatter(wall_points_2d[:, 0], wall_points_2d[:, 1], s=8, c='red', alpha=0.7, label='Wall points')
        if walls:
            for i, wall in enumerate(walls):
                slope, intercept, inlier_points = wall
                x_min, x_max = np.min(inlier_points[:,0]), np.max(inlier_points[:,0])
                x_vals = np.array([x_min, x_max])
                y_vals = slope * x_vals + intercept
                ax2d.plot(x_vals, y_vals, '-', color='blue', linewidth=2, alpha=0.9, label='Wall segment' if i == 0 else None)
                ax2d.scatter(inlier_points[:, 0], inlier_points[:, 1], s=8, c='blue', alpha=0.8, label='Wall inliers' if i == 0 else None)
        handles, labels = ax2d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2d.legend(by_label.values(), by_label.keys())
        plt.pause(0.1)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
