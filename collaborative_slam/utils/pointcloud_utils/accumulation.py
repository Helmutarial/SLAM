"""
Accumulation & Merging

Funciones para acumular, fusionar y guardar nubes de puntos acumuladas.
"""
import open3d as o3d
import os
import numpy as np

def merge_point_clouds(clouds):
    """
    Merge multiple point clouds into one.
    Args:
        clouds: List of Open3D point clouds
    Returns:
        Merged point cloud
    """
    merged_cloud = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged_cloud += cloud
    return merged_cloud

def save_accumulated_clouds(accumulated_clouds, output_folder, frame_counter, camera_matrix=None):
    """
    Save accumulated point clouds to a PLY file.
    Args:
        accumulated_clouds: List of point clouds to merge and save
        output_folder: Output directory
        frame_counter: Frame number for filename
        camera_matrix: Camera transformation matrix (optional)
    Returns:
        Path to saved PLY file
    """
    merged_cloud = merge_point_clouds(accumulated_clouds)
    if camera_matrix is not None:
        merged_cloud.transform(camera_matrix)
    filename = os.path.join(output_folder, f"accumulated_{frame_counter:07d}.ply")
    o3d.io.write_point_cloud(filename, merged_cloud)
    print(f"✅ Saved accumulated cloud: {filename}")
    return filename
