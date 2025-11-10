"""
Visualize Trajectory Alignment - Step by Step

This script shows:
1. VIDEO2 original: trajectory + point clouds
2. VIDEO4 original: trajectory + point clouds  
3. Collaborative map: aligned trajectories + merged point clouds

This helps validate that the alignment is working correctly.

Functions:
    - extract_camera_positions: Get camera positions from keypoints
    - load_point_clouds: Load and combine point clouds from directory
    - create_trajectory_viz: Create trajectory visualization
    - show_video_original: Show individual video in original coordinates
    - show_collaborative: Show aligned collaborative map
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import argparse


def extract_camera_positions(keypoint_dir):
    """Extract camera positions from cam_to_world matrices."""
    positions = []
    
    for kpt_file in sorted(keypoint_dir.glob("keypoints_*.npz")):
        data = np.load(kpt_file)
        cam_to_world = data['cam_to_world']
        cam_pos = cam_to_world[:3, 3]
        positions.append(cam_pos)
    
    return np.array(positions)


def load_point_clouds(cloud_dir, max_frames=10, color=None):
    """Load and combine point clouds from directory."""
    cloud_files = sorted(Path(cloud_dir).glob("*.ply"))[:max_frames]
    
    combined = o3d.geometry.PointCloud()
    for ply_file in cloud_files:
        pcd = o3d.io.read_point_cloud(str(ply_file))
        if color:
            pcd.paint_uniform_color(color)
        combined += pcd
    
    return combined


def create_trajectory_viz(positions, color, size=0.02):
    """Create trajectory visualization with line and spheres."""
    geometries = []
    
    # Line connecting positions
    points = o3d.utility.Vector3dVector(positions)
    lines = [[i, i+1] for i in range(len(positions)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    geometries.append(line_set)
    
    # Sphere markers
    for i in range(0, len(positions), 3):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(positions[i])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    return geometries


def transform_points(points, transform_matrix):
    """Apply 4x4 transformation to 3D points."""
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])
    transformed_homo = (transform_matrix @ points_homo.T).T
    return transformed_homo[:, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoint_dir1", type=str, required=True)
    parser.add_argument("--keypoint_dir2", type=str, required=True)
    parser.add_argument("--match_file", type=str, required=True)
    parser.add_argument("--collab_map", type=str, required=True)
    
    args = parser.parse_args()
    
    keypoint_dir1 = Path(args.keypoint_dir1)
    keypoint_dir2 = Path(args.keypoint_dir2)
    cloud_dir1 = keypoint_dir1.parent / "cloud_points"
    cloud_dir2 = keypoint_dir2.parent / "cloud_points"
    
    # Extract trajectories
    print("Extracting camera trajectories...")
    traj1 = extract_camera_positions(keypoint_dir1)
    traj2 = extract_camera_positions(keypoint_dir2)
    
    print(f"VIDEO2: {len(traj1)} positions")
    print(f"VIDEO4: {len(traj2)} positions")
    
    # ========== STEP 1: VIDEO2 ORIGINAL ==========
    print("\n" + "="*70)
    print("STEP 1: VIDEO2 - Original Coordinates")
    print("="*70)
    
    print("Loading VIDEO2 point clouds (first 10 frames)...")
    clouds_v2 = load_point_clouds(cloud_dir1, max_frames=10, color=[1, 0.3, 0.3])
    
    pts = np.asarray(clouds_v2.points)
    print(f"Point cloud: {len(pts):,} points")
    print(f"  Range: X[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    print(f"Trajectory: {len(traj1)} positions")
    print(f"  Range: X[{traj1[:,0].min():.2f}, {traj1[:,0].max():.2f}] "
          f"Y[{traj1[:,1].min():.2f}, {traj1[:,1].max():.2f}] "
          f"Z[{traj1[:,2].min():.2f}, {traj1[:,2].max():.2f}]")
    
    traj1_viz = create_trajectory_viz(traj1, color=[1, 0, 0], size=0.05)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    print("\n🔴 Red trajectory should pass THROUGH red point cloud")
    print("📍 Close window to continue to next step...")
    
    o3d.visualization.draw_geometries(
        [clouds_v2, coord_frame] + traj1_viz,
        window_name="STEP 1: VIDEO2 - Original",
        width=1920,
        height=1080
    )
    
    # ========== STEP 2: VIDEO4 ORIGINAL ==========
    print("\n" + "="*70)
    print("STEP 2: VIDEO4 - Original Coordinates")
    print("="*70)
    
    print("Loading VIDEO4 point clouds (first 10 frames)...")
    clouds_v4 = load_point_clouds(cloud_dir2, max_frames=10, color=[0.3, 0.3, 1])
    
    pts = np.asarray(clouds_v4.points)
    print(f"Point cloud: {len(pts):,} points")
    print(f"  Range: X[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    print(f"Trajectory: {len(traj2)} positions")
    print(f"  Range: X[{traj2[:,0].min():.2f}, {traj2[:,0].max():.2f}] "
          f"Y[{traj2[:,1].min():.2f}, {traj2[:,1].max():.2f}] "
          f"Z[{traj2[:,2].min():.2f}, {traj2[:,2].max():.2f}]")
    
    traj2_viz = create_trajectory_viz(traj2, color=[0, 0, 1], size=0.05)
    
    print("\n🔵 Blue trajectory should pass THROUGH blue point cloud")
    print("📍 Close window to continue to next step...")
    
    o3d.visualization.draw_geometries(
        [clouds_v4, coord_frame] + traj2_viz,
        window_name="STEP 2: VIDEO4 - Original",
        width=1920,
        height=1080
    )
    
    # ========== STEP 3: COLLABORATIVE MAP ==========
    print("\n" + "="*70)
    print("STEP 3: Collaborative Map - Aligned Coordinates")
    print("="*70)
    
    # Load match
    with open(args.match_file, 'r') as f:
        matches = json.load(f)
    
    best_match = matches[0]
    print(f"Using alignment from frame {best_match['video1_frame']} <-> {best_match['video2_frame']}")
    
    # Load transformation matrices
    kpts1_data = np.load(keypoint_dir1 / f"keypoints_{best_match['video1_frame']}.npz")
    kpts2_data = np.load(keypoint_dir2 / f"keypoints_{best_match['video2_frame']}.npz")
    
    cam_to_world_1 = kpts1_data['cam_to_world']
    cam_to_world_2 = kpts2_data['cam_to_world']
    
    world_to_cam_1 = np.linalg.inv(cam_to_world_1)
    world_to_cam_2 = np.linalg.inv(cam_to_world_2)
    
    usip_transformation = np.array(best_match['transformation'])
    
    # Transform trajectories
    print("Applying transformations to trajectories...")
    traj1_cam = transform_points(traj1, world_to_cam_1)
    alignment_transform = world_to_cam_2 @ usip_transformation @ cam_to_world_1
    traj1_aligned = transform_points(traj1_cam, alignment_transform)
    
    traj2_cam = transform_points(traj2, world_to_cam_2)
    
    # Load collaborative map
    print("Loading collaborative map...")
    collab_map = o3d.io.read_point_cloud(args.collab_map)
    
    pts = np.asarray(collab_map.points)
    print(f"Collaborative map: {len(pts):,} points")
    print(f"  Range: X[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    
    print(f"Traj1 aligned: {len(traj1_aligned)} positions")
    print(f"  Range: X[{traj1_aligned[:,0].min():.2f}, {traj1_aligned[:,0].max():.2f}] "
          f"Y[{traj1_aligned[:,1].min():.2f}, {traj1_aligned[:,1].max():.2f}] "
          f"Z[{traj1_aligned[:,2].min():.2f}, {traj1_aligned[:,2].max():.2f}]")
    
    print(f"Traj2 transformed: {len(traj2_cam)} positions")
    print(f"  Range: X[{traj2_cam[:,0].min():.2f}, {traj2_cam[:,0].max():.2f}] "
          f"Y[{traj2_cam[:,1].min():.2f}, {traj2_cam[:,1].max():.2f}] "
          f"Z[{traj2_cam[:,2].min():.2f}, {traj2_cam[:,2].max():.2f}]")
    
    traj1_viz_aligned = create_trajectory_viz(traj1_aligned, color=[1, 0, 0], size=0.02)
    traj2_viz_aligned = create_trajectory_viz(traj2_cam, color=[0, 0, 1], size=0.02)
    
    print("\n🔴 Red trajectory (VIDEO2 aligned)")
    print("🔵 Blue trajectory (VIDEO4)")
    print("📍 Both should pass through their respective colored point clouds")
    
    o3d.visualization.draw_geometries(
        [collab_map, coord_frame] + traj1_viz_aligned + traj2_viz_aligned,
        window_name="STEP 3: Collaborative Map - Aligned",
        width=1920,
        height=1080
    )


if __name__ == "__main__":
    main()
