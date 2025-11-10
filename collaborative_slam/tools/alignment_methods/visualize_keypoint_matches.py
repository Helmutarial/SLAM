"""
Visualize Keypoint Matches

This script visualizes matched keypoints between consecutive frames
to validate alignment quality and inspect matching results.

Functions:
    - visualize_matches: Display matched keypoints between two frames
    - visualize_transformation: Show before/after alignment of point clouds
    - create_match_report: Generate HTML report with match statistics
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def load_point_cloud(ply_file):
    """Load point cloud from PLY file."""
    pcd = o3d.io.read_point_cloud(str(ply_file))
    return pcd


def visualize_matches(keypoint_dir, frame1_id, frame2_id, matching_results, 
                     cloud_dir=None, show_clouds=True):
    """
    Visualize matched keypoints between two frames.
    
    Args:
        keypoint_dir (str): Directory containing keypoint files
        frame1_id (str): Source frame ID
        frame2_id (str): Target frame ID
        matching_results (dict): Matching results dictionary
        cloud_dir (str): Optional directory containing original point clouds
        show_clouds (bool): Whether to show full point clouds or just keypoints
    """
    keypoint_dir = Path(keypoint_dir)
    
    # Load keypoints
    kpts1_file = keypoint_dir / f"keypoints_{frame1_id}.npz"
    kpts2_file = keypoint_dir / f"keypoints_{frame2_id}.npz"
    
    kpts1_data = np.load(kpts1_file)
    kpts2_data = np.load(kpts2_file)
    
    kpts1_world = kpts1_data['keypoints_world']
    kpts2_world = kpts2_data['keypoints_world']
    
    # Get matching result
    pair_key = f"{frame1_id}_{frame2_id}"
    match_result = matching_results.get(pair_key)
    
    if match_result is None:
        print(f"No matching result found for {pair_key}")
        return
    
    # Create visualization
    geometries = []
    
    if show_clouds and cloud_dir is not None:
        # Load full point clouds
        cloud_dir = Path(cloud_dir)
        pcd1 = load_point_cloud(cloud_dir / f"{frame1_id}.ply")
        pcd2 = load_point_cloud(cloud_dir / f"{frame2_id}.ply")
        
        # Color source cloud red, target cloud blue
        pcd1.paint_uniform_color([1, 0, 0])
        pcd2.paint_uniform_color([0, 0, 1])
        
        geometries.extend([pcd1, pcd2])
    else:
        # Create point clouds from keypoints only
        pcd1_kpts = o3d.geometry.PointCloud()
        pcd2_kpts = o3d.geometry.PointCloud()
        pcd1_kpts.points = o3d.utility.Vector3dVector(kpts1_world)
        pcd2_kpts.points = o3d.utility.Vector3dVector(kpts2_world)
        
        pcd1_kpts.paint_uniform_color([1, 0, 0])  # Red for source
        pcd2_kpts.paint_uniform_color([0, 0, 1])  # Blue for target
        
        geometries.extend([pcd1_kpts, pcd2_kpts])
    
    # Create lines connecting matched keypoints
    transformation = np.array(match_result['transformation'])
    
    # Get inlier matches (reconstruct from results)
    # Note: We don't save individual match indices, so we'll visualize 
    # by showing transformed source keypoints
    
    # Transform source keypoints
    pcd1_transformed = o3d.geometry.PointCloud()
    pcd1_transformed.points = o3d.utility.Vector3dVector(kpts1_world)
    pcd1_transformed.transform(transformation)
    pcd1_transformed.paint_uniform_color([0, 1, 0])  # Green for transformed
    
    geometries.append(pcd1_transformed)
    
    # Visualize
    print(f"\nVisualization for frames {frame1_id} -> {frame2_id}")
    print(f"Inliers: {match_result['num_inliers']}/{match_result['num_matches']}")
    print(f"Inlier ratio: {match_result['inlier_ratio']:.2%}")
    print(f"RMSE: {match_result['rmse']:.4f}m")
    print("\nColors:")
    print("  Red: Source keypoints (frame {})".format(frame1_id))
    print("  Blue: Target keypoints (frame {})".format(frame2_id))
    print("  Green: Transformed source keypoints")
    
    o3d.visualization.draw_geometries(geometries, 
                                     window_name=f"Matches {frame1_id} -> {frame2_id}")


def visualize_transformation_quality(keypoint_dir, frame1_id, frame2_id, 
                                    matching_results, cloud_dir):
    """
    Visualize alignment quality by showing before/after transformation.
    
    Args:
        keypoint_dir (str): Directory containing keypoint files
        frame1_id (str): Source frame ID
        frame2_id (str): Target frame ID
        matching_results (dict): Matching results dictionary
        cloud_dir (str): Directory containing original point clouds
    """
    cloud_dir = Path(cloud_dir)
    keypoint_dir = Path(keypoint_dir)
    
    # Load keypoint data to get camera-to-world transformations
    kpts1_file = keypoint_dir / f"keypoints_{frame1_id}.npz"
    kpts2_file = keypoint_dir / f"keypoints_{frame2_id}.npz"
    
    kpts1_data = np.load(kpts1_file)
    kpts2_data = np.load(kpts2_file)
    
    cam_to_world_1 = kpts1_data['cam_to_world']
    cam_to_world_2 = kpts2_data['cam_to_world']
    
    # Load full point clouds (these are in world coordinates already)
    pcd1_world = load_point_cloud(cloud_dir / f"{frame1_id}.ply")
    pcd2_world = load_point_cloud(cloud_dir / f"{frame2_id}.ply")
    
    # Convert to camera coordinates by applying inverse transformation
    world_to_cam_1 = np.linalg.inv(cam_to_world_1)
    world_to_cam_2 = np.linalg.inv(cam_to_world_2)
    
    pcd1_cam = o3d.geometry.PointCloud(pcd1_world)
    pcd2_cam = o3d.geometry.PointCloud(pcd2_world)
    pcd1_cam.transform(world_to_cam_1)
    pcd2_cam.transform(world_to_cam_2)
    
    # Get USIP transformation (from frame1 to frame2 in world coordinates)
    pair_key = f"{frame1_id}_{frame2_id}"
    match_result = matching_results.get(pair_key)
    
    if match_result is None:
        print(f"No matching result found for {pair_key}")
        return
    
    usip_transformation = np.array(match_result['transformation'])
    
    # BEFORE: Show both clouds in their own camera coordinates (misaligned)
    pcd1_before = o3d.geometry.PointCloud(pcd1_cam)
    pcd2_before = o3d.geometry.PointCloud(pcd2_cam)
    pcd1_before.paint_uniform_color([1, 0, 0])  # Red
    pcd2_before.paint_uniform_color([0, 0, 1])  # Blue
    
    print(f"\n{'='*60}")
    print(f"BEFORE USIP Alignment")
    print(f"{'='*60}")
    print(f"Frames {frame1_id} (red) and {frame2_id} (blue) in their own camera coordinates")
    print("These clouds are NOT aligned - they show different viewpoints")
    o3d.visualization.draw_geometries([pcd1_before, pcd2_before],
                                     window_name="BEFORE - Misaligned Camera Frames")
    
    # AFTER: Transform frame1 to align with frame2 using USIP
    # We need to transform from world1 to world2
    # The USIP transformation is: T_usip * points_world1 = points_world2
    # But we want to see in camera2 frame, so:
    # cam2_points = world_to_cam_2 @ usip_transformation @ cam_to_world_1 @ cam1_points
    
    alignment_transform = world_to_cam_2 @ usip_transformation @ cam_to_world_1
    
    pcd1_after = o3d.geometry.PointCloud(pcd1_cam)
    pcd2_after = o3d.geometry.PointCloud(pcd2_cam)
    pcd1_after.transform(alignment_transform)
    pcd1_after.paint_uniform_color([0, 1, 0])  # Green
    pcd2_after.paint_uniform_color([0, 0, 1])  # Blue
    
    print(f"\n{'='*60}")
    print(f"AFTER USIP Alignment")
    print(f"{'='*60}")
    print(f"Frame {frame1_id} transformed (green) aligned with frame {frame2_id} (blue)")
    print(f"RMSE: {match_result['rmse']:.4f}m ({match_result['rmse']*100:.2f}cm)")
    print(f"Inliers: {match_result['num_inliers']}/{match_result['num_matches']} ({match_result['inlier_ratio']:.1%})")
    print("\nGreen and blue clouds should now overlap significantly")
    o3d.visualization.draw_geometries([pcd1_after, pcd2_after],
                                     window_name="AFTER - USIP Aligned")


def plot_matching_statistics(matching_results, output_file=None):
    """
    Plot statistics about matching results.
    
    Args:
        matching_results (dict): Matching results dictionary
        output_file (str): Optional file to save plot
    """
    # Extract statistics
    successful = []
    inlier_ratios = []
    rmse_values = []
    num_matches_list = []
    
    for pair_id, result in matching_results.items():
        if result is not None:
            successful.append(True)
            inlier_ratios.append(result['inlier_ratio'])
            rmse_values.append(result['rmse'])
            num_matches_list.append(result['num_matches'])
        else:
            successful.append(False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('USIP Keypoint Matching Statistics', fontsize=16)
    
    # Plot 1: Success rate
    ax = axes[0, 0]
    success_rate = sum(successful) / len(successful) * 100
    ax.bar(['Successful', 'Failed'], 
           [sum(successful), len(successful) - sum(successful)],
           color=['green', 'red'])
    ax.set_ylabel('Count')
    ax.set_title(f'Match Success Rate: {success_rate:.1f}%')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Inlier ratio distribution
    ax = axes[0, 1]
    ax.hist(inlier_ratios, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(inlier_ratios), color='red', linestyle='--', 
               label=f'Mean: {np.mean(inlier_ratios):.2%}')
    ax.set_xlabel('Inlier Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Inlier Ratio Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: RMSE distribution
    ax = axes[1, 0]
    ax.hist(rmse_values, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rmse_values), color='red', linestyle='--',
               label=f'Mean: {np.mean(rmse_values):.4f}m')
    ax.set_xlabel('RMSE (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('RMSE Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Number of matches
    ax = axes[1, 1]
    ax.hist(num_matches_list, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(num_matches_list), color='red', linestyle='--',
               label=f'Mean: {np.mean(num_matches_list):.1f}')
    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Frequency')
    ax.set_title('Initial Matches Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize keypoint matching results")
    parser.add_argument("--keypoint_dir", type=str, required=True,
                       help="Directory containing keypoint files")
    parser.add_argument("--matching_results", type=str, required=True,
                       help="JSON file with matching results")
    parser.add_argument("--cloud_dir", type=str, default=None,
                       help="Directory containing original point clouds")
    parser.add_argument("--frame1", type=str, default=None,
                       help="Source frame ID for visualization")
    parser.add_argument("--frame2", type=str, default=None,
                       help="Target frame ID for visualization")
    parser.add_argument("--plot_stats", action="store_true",
                       help="Plot matching statistics")
    parser.add_argument("--plot_output", type=str, default=None,
                       help="Output file for statistics plot")
    parser.add_argument("--show_alignment", action="store_true",
                       help="Show before/after alignment comparison")
    
    args = parser.parse_args()
    
    # Load matching results
    with open(args.matching_results, 'r') as f:
        matching_results = json.load(f)
    
    # Plot statistics if requested
    if args.plot_stats:
        plot_matching_statistics(matching_results, args.plot_output)
    
    # Visualize specific frame pair if requested
    if args.frame1 and args.frame2:
        if args.show_alignment and args.cloud_dir:
            visualize_transformation_quality(
                args.keypoint_dir, args.frame1, args.frame2,
                matching_results, args.cloud_dir
            )
        else:
            visualize_matches(
                args.keypoint_dir, args.frame1, args.frame2,
                matching_results, args.cloud_dir,
                show_clouds=(args.cloud_dir is not None)
            )
