"""
Preprocessing & Filtering

Funciones para preprocesar, filtrar y limpiar nubes de puntos.
"""
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsample the point cloud and extract FPFH features.
    Args:
        pcd: Open3D point cloud object.
        voxel_size: Voxel size for downsampling.
    Returns:
        Tuple (downsampled point cloud, FPFH features)
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def filter_isolated_points(points, z, percentile=90, n_neighbors=6):
    """
    Remove isolated points from a point cloud using neighbor distance filtering.
    Args:
        points (np.ndarray): Nx2 array of XY points
        z (np.ndarray): N array of Z values
        percentile (float): Percentile threshold for isolation
        n_neighbors (int): Number of neighbors to consider
    Returns:
        Tuple (filtered_points, filtered_z)
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_dist = distances[:, 1:].mean(axis=1)
    noise_mask = mean_dist < np.percentile(mean_dist, percentile)
    return points[noise_mask], z[noise_mask]

def filter_points_by_percentile(points, z, x_percentile=[2,98], y_percentile=[2,98]):
    """
    Filter points by X and Y percentiles to center visualization.
    Args:
        points (np.ndarray): Nx2 array of XY points
        z (np.ndarray): N array of Z values
        x_percentile (list): Percentiles for X axis
        y_percentile (list): Percentiles for Y axis
    Returns:
        Tuple (filtered_points, filtered_z)
    """
    x_min, x_max = np.percentile(points[:, 0], x_percentile)
    y_min, y_max = np.percentile(points[:, 1], y_percentile)
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    return points[mask], z[mask]