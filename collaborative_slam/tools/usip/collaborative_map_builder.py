"""
Collaborative Map Builder

This script builds a unified 3D map by merging point clouds from multiple
cameras using USIP-based cross-camera alignment transformations.

Functions:
    - load_cross_camera_matches: Load alignment results
    - merge_point_clouds: Combine clouds from different cameras
    - build_collaborative_map: Create unified map from multiple videos
    - visualize_collaborative_map: Display merged map with camera-coded colors
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import json
import argparse


def load_cross_camera_matches(match_file):
    """Load cross-camera matching results."""
    with open(match_file, 'r') as f:
        matches = json.load(f)
    return matches


def merge_point_clouds_from_match(cloud_dir1, cloud_dir2, frame1_id, frame2_id,
                                  transformation, keypoint_dir1, keypoint_dir2,
                                  color1=[1, 0, 0], color2=[0, 0, 1]):
    """
    Merge two point clouds from different cameras using computed transformation.
    
    Args:
        cloud_dir1 (str): Point cloud directory for camera 1
        cloud_dir2 (str): Point cloud directory for camera 2
        frame1_id (str): Frame ID from camera 1
        frame2_id (str): Frame ID from camera 2
        transformation (np.ndarray): 4x4 transformation matrix
        keypoint_dir1 (str): Keypoint directory for camera 1
        keypoint_dir2 (str): Keypoint directory for camera 2
        color1 (list): RGB color for camera 1 points
        color2 (list): RGB color for camera 2 points
        
    Returns:
        o3d.geometry.PointCloud: Merged point cloud
    """
    cloud_dir1 = Path(cloud_dir1)
    cloud_dir2 = Path(cloud_dir2)
    keypoint_dir1 = Path(keypoint_dir1)
    keypoint_dir2 = Path(keypoint_dir2)
    
    # Load keypoint data to get camera-to-world transforms
    kpts1_data = np.load(keypoint_dir1 / f"keypoints_{frame1_id}.npz")
    kpts2_data = np.load(keypoint_dir2 / f"keypoints_{frame2_id}.npz")
    
    cam_to_world_1 = kpts1_data['cam_to_world']
    cam_to_world_2 = kpts2_data['cam_to_world']
    
    # Load point clouds (in world coordinates)
    pcd1_world = o3d.io.read_point_cloud(str(cloud_dir1 / f"{frame1_id}.ply"))
    pcd2_world = o3d.io.read_point_cloud(str(cloud_dir2 / f"{frame2_id}.ply"))
    
    # Convert to camera coordinates
    world_to_cam_1 = np.linalg.inv(cam_to_world_1)
    world_to_cam_2 = np.linalg.inv(cam_to_world_2)
    
    pcd1_cam = o3d.geometry.PointCloud(pcd1_world)
    pcd2_cam = o3d.geometry.PointCloud(pcd2_world)
    pcd1_cam.transform(world_to_cam_1)
    pcd2_cam.transform(world_to_cam_2)
    
    # Transform camera 1 to align with camera 2 coordinate system
    alignment_transform = world_to_cam_2 @ transformation @ cam_to_world_1
    pcd1_aligned = o3d.geometry.PointCloud(pcd1_cam)
    pcd1_aligned.transform(alignment_transform)
    
    # Color the point clouds
    pcd1_aligned.paint_uniform_color(color1)
    pcd2_cam.paint_uniform_color(color2)
    
    # Merge point clouds
    merged = pcd1_aligned + pcd2_cam
    
    return merged


def build_collaborative_map(match_file, cloud_dir1, cloud_dir2, 
                           keypoint_dir1, keypoint_dir2, num_frames=5):
    """
    Build collaborative map using multiple matched frame pairs.
    
    Args:
        match_file (str): JSON file with cross-camera matches
        cloud_dir1 (str): Point cloud directory for camera 1
        cloud_dir2 (str): Point cloud directory for camera 2
        keypoint_dir1 (str): Keypoint directory for camera 1
        keypoint_dir2 (str): Keypoint directory for camera 2
        num_frames (int): Number of frame pairs to include
        
    Returns:
        o3d.geometry.PointCloud: Collaborative map
    """
    matches = load_cross_camera_matches(match_file)
    
    # Sort by RMSE (best quality first)
    matches.sort(key=lambda x: x['rmse'])
    
    # Use top num_frames matches
    selected_matches = matches[:min(num_frames, len(matches))]
    
    print(f"Building collaborative map with {len(selected_matches)} frame pairs")
    
    # Colors for visualization
    camera1_color = [1, 0.3, 0.3]  # Light red
    camera2_color = [0.3, 0.3, 1]  # Light blue
    
    all_clouds = []
    
    for i, match in enumerate(selected_matches):
        print(f"\nMerging pair {i+1}/{len(selected_matches)}")
        print(f"  Camera 1 frame {match['video1_frame']} <-> Camera 2 frame {match['video2_frame']}")
        print(f"  RMSE: {match['rmse']:.4f}m, Inlier ratio: {match['inlier_ratio']:.2%}")
        
        transformation = np.array(match['transformation'])
        
        merged = merge_point_clouds_from_match(
            cloud_dir1, cloud_dir2,
            match['video1_frame'], match['video2_frame'],
            transformation,
            keypoint_dir1, keypoint_dir2,
            color1=camera1_color,
            color2=camera2_color
        )
        
        all_clouds.append(merged)
    
    # Combine all merged clouds
    collaborative_map = o3d.geometry.PointCloud()
    for cloud in all_clouds:
        collaborative_map += cloud
    
    # Remove statistical outliers
    collaborative_map, _ = collaborative_map.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    
    return collaborative_map


def visualize_collaborative_map(collaborative_map, title="Collaborative SLAM Map"):
    """
    Visualize the collaborative map.
    
    Args:
        collaborative_map (o3d.geometry.PointCloud): Merged point cloud
        title (str): Window title
    """
    print(f"\n{'='*60}")
    print(f"COLLABORATIVE SLAM MAP")
    print(f"{'='*60}")
    print(f"Total points: {len(collaborative_map.points):,}")
    print(f"\nColor coding:")
    print(f"  Light Red: Camera 1 (VIDEO2)")
    print(f"  Light Blue: Camera 2 (VIDEO4)")
    print(f"\nOverlapping areas show multi-camera coverage")
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries(
        [collaborative_map, coordinate_frame],
        window_name=title,
        width=1280,
        height=720
    )


def save_collaborative_map(collaborative_map, output_file):
    """Save collaborative map to PLY file."""
    o3d.io.write_point_cloud(output_file, collaborative_map)
    print(f"\nSaved collaborative map to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build collaborative SLAM map from multiple cameras")
    parser.add_argument("--match_file", type=str, required=True,
                       help="JSON file with cross-camera matches")
    parser.add_argument("--cloud_dir1", type=str, required=True,
                       help="Point cloud directory for camera 1")
    parser.add_argument("--cloud_dir2", type=str, required=True,
                       help="Point cloud directory for camera 2")
    parser.add_argument("--keypoint_dir1", type=str, required=True,
                       help="Keypoint directory for camera 1")
    parser.add_argument("--keypoint_dir2", type=str, required=True,
                       help="Keypoint directory for camera 2")
    parser.add_argument("--num_frames", type=int, default=5,
                       help="Number of frame pairs to include in map")
    parser.add_argument("--output", type=str, default=None,
                       help="Output PLY file for collaborative map")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize the collaborative map")
    
    args = parser.parse_args()
    
    # Build collaborative map
    collaborative_map = build_collaborative_map(
        args.match_file,
        args.cloud_dir1,
        args.cloud_dir2,
        args.keypoint_dir1,
        args.keypoint_dir2,
        num_frames=args.num_frames
    )
    
    # Save if requested
    if args.output:
        save_collaborative_map(collaborative_map, args.output)
    
    # Visualize if requested
    if args.visualize:
        visualize_collaborative_map(collaborative_map)
