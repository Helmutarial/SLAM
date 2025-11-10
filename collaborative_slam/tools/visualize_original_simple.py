"""
Simple Visualization - Original Coordinates Only

Show point clouds and camera trajectories in their ORIGINAL coordinates
without any transformations. This is to verify the data is correct.

Functions:
    - extract_camera_positions: Get camera positions from keypoints
    - visualize_video: Show one video's clouds + trajectory
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import argparse


def extract_camera_positions(keypoint_dir):
    """
    Extract camera positions from cam_to_world matrices in keypoint files.
    
    Args:
        keypoint_dir (Path): Directory with keypoint .npz files
        
    Returns:
        np.ndarray: Nx3 array of camera positions
    """
    positions = []
    frame_ids = []
    
    keypoint_dir = Path(keypoint_dir)
    for kpt_file in sorted(keypoint_dir.glob("keypoints_*.npz")):
        frame_id = int(kpt_file.stem.split('_')[1])
        data = np.load(kpt_file)
        cam_to_world = data['cam_to_world']
        cam_pos = cam_to_world[:3, 3]
        
        positions.append(cam_pos)
        frame_ids.append(frame_id)
    
    return np.array(positions), frame_ids


def create_trajectory_line(positions, color=[1, 0, 0]):
    """
    Create Open3D LineSet for trajectory.
    
    Args:
        positions (np.ndarray): Nx3 camera positions
        color (list): RGB color
        
    Returns:
        o3d.geometry.LineSet: Trajectory line
    """
    points = o3d.utility.Vector3dVector(positions)
    lines = [[i, i+1] for i in range(len(positions)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return line_set


def create_camera_spheres(positions, color=[1, 0, 0], size=0.05):
    """
    Create sphere markers at camera positions.
    
    Args:
        positions (np.ndarray): Nx3 camera positions
        color (list): RGB color
        size (float): Sphere radius
        
    Returns:
        list: List of sphere geometries
    """
    spheres = []
    for i in range(0, len(positions), 3):  # Every 3rd position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(positions[i])
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    
    return spheres


def load_point_clouds(cloud_dir, frame_ids=None, max_clouds=None, color=None):
    """
    Load point clouds from directory.
    
    Args:
        cloud_dir (Path): Directory with .ply files
        frame_ids (list): Specific frame IDs to load (None = all)
        max_clouds (int): Maximum number of clouds to load
        color (list): RGB color to paint clouds
        
    Returns:
        o3d.geometry.PointCloud: Combined point cloud
    """
    cloud_dir = Path(cloud_dir)
    combined = o3d.geometry.PointCloud()
    
    if frame_ids is not None:
        # Load specific frames
        for frame_id in frame_ids[:max_clouds] if max_clouds else frame_ids:
            ply_file = cloud_dir / f"{frame_id}.ply"
            if ply_file.exists():
                pcd = o3d.io.read_point_cloud(str(ply_file))
                if color:
                    pcd.paint_uniform_color(color)
                combined += pcd
    else:
        # Load all clouds
        cloud_files = sorted(cloud_dir.glob("*.ply"))
        if max_clouds:
            cloud_files = cloud_files[:max_clouds]
        
        for ply_file in cloud_files:
            pcd = o3d.io.read_point_cloud(str(ply_file))
            if color:
                pcd.paint_uniform_color(color)
            combined += pcd
    
    return combined


def visualize_video(keypoint_dir, cloud_dir, video_name, color, max_clouds=10):
    """
    Visualize one video's point clouds and camera trajectory.
    
    Args:
        keypoint_dir (str): Directory with keypoint files
        cloud_dir (str): Directory with point cloud files
        video_name (str): Name for display
        color (list): RGB color [r, g, b]
        max_clouds (int): Maximum clouds to load
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZING: {video_name}")
    print(f"{'='*70}")
    
    # Extract camera positions
    print("Extracting camera positions from keypoints...")
    keypoint_dir = Path(keypoint_dir)
    camera_positions, frame_ids = extract_camera_positions(keypoint_dir)
    
    print(f"Camera trajectory: {len(camera_positions)} positions")
    print(f"  X range: [{camera_positions[:,0].min():.3f}, {camera_positions[:,0].max():.3f}]")
    print(f"  Y range: [{camera_positions[:,1].min():.3f}, {camera_positions[:,1].max():.3f}]")
    print(f"  Z range: [{camera_positions[:,2].min():.3f}, {camera_positions[:,2].max():.3f}]")
    
    # Load point clouds
    print(f"\nLoading point clouds (max {max_clouds} frames)...")
    cloud_dir = Path(cloud_dir)
    point_cloud = load_point_clouds(
        cloud_dir, 
        frame_ids=frame_ids,
        max_clouds=max_clouds,
        color=color
    )
    
    points = np.asarray(point_cloud.points)
    print(f"Point cloud: {len(points):,} points")
    print(f"  X range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"  Y range: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"  Z range: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # Create trajectory visualization
    print("\nCreating trajectory visualization...")
    traj_line = create_trajectory_line(camera_positions, color=color)
    camera_spheres = create_camera_spheres(camera_positions, color=color, size=0.05)
    
    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Combine geometries
    geometries = [point_cloud, traj_line, coord_frame] + camera_spheres
    
    # Visualize
    print(f"\n✅ Visualization ready!")
    print(f"Expected: Trajectory (line + spheres) should pass THROUGH point cloud")
    print(f"Close window to continue...")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{video_name} - Original Coordinates",
        width=1920,
        height=1080
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds and trajectories in original coordinates")
    parser.add_argument("--keypoint_dir", type=str, required=True,
                       help="Directory with keypoint .npz files")
    parser.add_argument("--cloud_dir", type=str, required=True,
                       help="Directory with point cloud .ply files")
    parser.add_argument("--video_name", type=str, default="Video",
                       help="Name for display")
    parser.add_argument("--color", type=float, nargs=3, default=[1.0, 0.3, 0.3],
                       help="RGB color for visualization (3 values between 0-1)")
    parser.add_argument("--max_clouds", type=int, default=10,
                       help="Maximum number of point clouds to load")
    
    args = parser.parse_args()
    
    visualize_video(
        args.keypoint_dir,
        args.cloud_dir,
        args.video_name,
        args.color,
        args.max_clouds
    )


if __name__ == "__main__":
    main()
