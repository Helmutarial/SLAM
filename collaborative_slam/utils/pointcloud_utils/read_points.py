"""
Point Cloud Spatial Analysis Tool

This module provides tools for analyzing spatial boundaries and statistics
of point cloud datasets.
"""

import open3d as o3d
import numpy as np
import os
import re


def extract_number(filename):
    """
    Extract number from filename for proper sorting.
    
    Args:
        filename (str): Filename to extract number from
        
    Returns:
        int: Extracted number or infinity if no number found
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def load_point_clouds(folder_path):
    """
    Load all point clouds from a folder.
    
    Args:
        folder_path (str): Path to folder containing .ply files
        
    Returns:
        tuple: (point_clouds, filenames)
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return [], []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    files = sorted(files, key=extract_number)
    
    if not files:
        print(f"❌ No .ply files found in {folder_path}")
        return [], []
    
    point_clouds = []
    valid_files = []
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if pcd.has_points():
                point_clouds.append(pcd)
                valid_files.append(filename)
            else:
                print(f"⚠️ {filename} is empty, skipping...")
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
    
    return point_clouds, valid_files


def analyze_spatial_bounds(folder_path):
    """
    Analyze spatial boundaries of all point clouds in a folder.
    
    Args:
        folder_path (str): Path to folder containing .ply files
        
    Returns:
        dict: Spatial bounds information
    """
    point_clouds, filenames = load_point_clouds(folder_path)
    
    if not point_clouds:
        return {}
    
    print(f"📂 Processing {len(point_clouds)} point clouds in '{folder_path}'...")
    
    # Collect all coordinates
    all_x, all_y, all_z = [], [], []
    
    for pcd, filename in zip(point_clouds, filenames):
        points = np.asarray(pcd.points)
        
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
        all_z.extend(points[:, 2])
    
    # Calculate bounds
    bounds = {
        'x_min': min(all_x), 'x_max': max(all_x),
        'y_min': min(all_y), 'y_max': max(all_y),
        'z_min': min(all_z), 'z_max': max(all_z),
        'total_points': len(all_x),
        'num_clouds': len(point_clouds)
    }
    
    # Display results
    print("\n📊 Spatial bounds analysis:")
    print(f"   X: {bounds['x_min']:.3f} to {bounds['x_max']:.3f}")
    print(f"   Y: {bounds['y_min']:.3f} to {bounds['y_max']:.3f}")
    print(f"   Z: {bounds['z_min']:.3f} to {bounds['z_max']:.3f}")
    print(f"   Total points: {bounds['total_points']:,}")
    print(f"   Point clouds: {bounds['num_clouds']}")
    
    return bounds


def main():
    """Main function for standalone execution."""
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "output1"  # Default folder
    
    print("📊 POINT CLOUD SPATIAL ANALYSIS")
    print("="*50)
    
    bounds = analyze_spatial_bounds(folder_path)
    
    if not bounds:
        print("❌ No valid point clouds found for analysis")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
