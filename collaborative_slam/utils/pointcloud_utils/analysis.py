"""
Analysis & Metrics

Funciones para analizar límites espaciales, calcular métricas y comparar nubes.
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_spatial_bounds(folder_path):
    """
    Analyze spatial boundaries of all point clouds in a folder.
    Args:
        folder_path (str): Path to folder containing .ply files
    Returns:
        dict: Spatial bounds information
    """
    from .cloud_file_manager import load_point_clouds
    point_clouds, filenames = load_point_clouds(folder_path)
    if not point_clouds:
        return {}
    print(f"📂 Processing {len(point_clouds)} point clouds in '{folder_path}'...")
    all_x, all_y, all_z = [], [], []
    for pcd, filename in zip(point_clouds, filenames):
        points = np.asarray(pcd.points)
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
        all_z.extend(points[:, 2])
    bounds = {
        'x_min': min(all_x), 'x_max': max(all_x),
        'y_min': min(all_y), 'y_max': max(all_y),
        'z_min': min(all_z), 'z_max': max(all_z),
        'total_points': len(all_x),
        'num_clouds': len(point_clouds)
    }
    print("\n📊 Spatial bounds analysis:")
    print(f"   X: {bounds['x_min']:.3f} to {bounds['x_max']:.3f}")
    print(f"   Y: {bounds['y_min']:.3f} to {bounds['y_max']:.3f}")
    print(f"   Z: {bounds['z_min']:.3f} to {bounds['z_max']:.3f}")
    print(f"   Total points: {bounds['total_points']:,}")
    print(f"   Point clouds: {bounds['num_clouds']}")
    return bounds
