"""
Cross-Camera Alignment for Collaborative SLAM

This script finds correspondences between point clouds from different cameras
viewing the same scene, enabling multi-camera SLAM and collaborative mapping.

Functions:
    - load_all_keypoints: Load keypoints from multiple videos
    - find_cross_camera_matches: Search for frames seeing same areas
    - estimate_camera_to_camera_transform: Calculate relative camera poses
    - build_collaborative_map: Merge point clouds from multiple cameras
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import json
import argparse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def load_all_keypoints(keypoint_dir):
    """
    Load all keypoints from a directory.
    
    Args:
        keypoint_dir (str): Directory containing keypoint .npz files
        
    Returns:
        dict: Dictionary mapping frame_id to keypoint data
    """
    keypoint_dir = Path(keypoint_dir)
    
    # Load summary
    summary_file = keypoint_dir / "keypoints_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    all_keypoints = {}
    for frame_id in summary['frames']:
        kpt_file = keypoint_dir / f"keypoints_{frame_id}.npz"
        data = np.load(kpt_file)
        all_keypoints[frame_id] = {
            'keypoints_camera': data['keypoints_camera'],
            'keypoints_world': data['keypoints_world'],
            'cam_to_world': data['cam_to_world']
        }
    
    return all_keypoints


def compute_overlap_score(kpts1_world, kpts2_world, max_distance=0.15):
    """
    Compute overlap score between two keypoint sets based on spatial proximity.
    
    Args:
        kpts1_world (np.ndarray): First keypoint set (N, 3)
        kpts2_world (np.ndarray): Second keypoint set (M, 3)
        max_distance (float): Maximum distance to consider points as overlapping
        
    Returns:
        float: Overlap score (0-1), higher means more overlap
    """
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(kpts2_world)
    distances, _ = nbrs.kneighbors(kpts1_world)
    
    # Count how many keypoints from kpts1 have a close match in kpts2
    close_matches = np.sum(distances < max_distance)
    overlap_ratio = close_matches / len(kpts1_world)
    
    return overlap_ratio


def find_cross_camera_matches(video1_keypoints, video2_keypoints, 
                              overlap_threshold=0.3, top_k=10):
    """
    Find frame pairs from different cameras that likely view the same scene.
    
    Args:
        video1_keypoints (dict): Keypoints from first video
        video2_keypoints (dict): Keypoints from second video
        overlap_threshold (float): Minimum overlap ratio to consider
        top_k (int): Return top K best matches
        
    Returns:
        list: List of tuples (video1_frame, video2_frame, overlap_score)
    """
    candidates = []
    
    print("Searching for cross-camera correspondences...")
    for frame1_id, kpts1_data in tqdm(video1_keypoints.items(), desc="Video1 frames"):
        kpts1_world = kpts1_data['keypoints_world']
        
        for frame2_id, kpts2_data in video2_keypoints.items():
            kpts2_world = kpts2_data['keypoints_world']
            
            # Compute bidirectional overlap
            overlap_1to2 = compute_overlap_score(kpts1_world, kpts2_world)
            overlap_2to1 = compute_overlap_score(kpts2_world, kpts1_world)
            
            # Use symmetric overlap (average)
            overlap_score = (overlap_1to2 + overlap_2to1) / 2.0
            
            if overlap_score >= overlap_threshold:
                candidates.append((frame1_id, frame2_id, overlap_score))
    
    # Sort by overlap score (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return candidates[:top_k]


def match_keypoints_cross_camera(kpts1_world, kpts2_world, max_distance=0.1):
    """
    Match keypoints between two cameras using spatial proximity.
    
    Args:
        kpts1_world (np.ndarray): Keypoints from camera 1 (N, 3)
        kpts2_world (np.ndarray): Keypoints from camera 2 (M, 3)
        max_distance (float): Maximum distance for valid match
        
    Returns:
        np.ndarray: Matches array (K, 2) with indices [idx1, idx2]
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(kpts2_world)
    distances, indices = nbrs.kneighbors(kpts1_world)
    
    # Filter by distance threshold
    matches = []
    for i in range(len(distances)):
        if distances[i][0] < max_distance:
            matches.append([i, indices[i][0]])
    
    return np.array(matches)


def estimate_camera_to_camera_transform(kpts1_world, kpts2_world, matches,
                                       ransac_threshold=0.1):
    """
    Estimate transformation between two camera coordinate systems.
    
    Args:
        kpts1_world (np.ndarray): Keypoints from camera 1 in world coords
        kpts2_world (np.ndarray): Keypoints from camera 2 in world coords
        matches (np.ndarray): Correspondence pairs
        ransac_threshold (float): RANSAC inlier threshold
        
    Returns:
        dict: Transformation result or None if failed
    """
    if len(matches) < 3:
        return None
    
    # Create point clouds from matched keypoints
    source_matched = o3d.geometry.PointCloud()
    target_matched = o3d.geometry.PointCloud()
    
    source_matched.points = o3d.utility.Vector3dVector(kpts1_world[matches[:, 0]])
    target_matched.points = o3d.utility.Vector3dVector(kpts2_world[matches[:, 1]])
    
    # Create correspondence set
    corr = np.array([[i, i] for i in range(len(matches))])
    correspondence_set = o3d.utility.Vector2iVector(corr)
    
    # RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=source_matched,
        target=target_matched,
        corres=correspondence_set,
        max_correspondence_distance=ransac_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
    )
    
    # Compute metrics
    inlier_indices = np.asarray(result.correspondence_set)
    inlier_ratio = len(inlier_indices) / len(matches)
    
    # Get inlier matches
    inlier_matches = matches[inlier_indices]
    
    # Get corresponding source and target points
    source_points = kpts1_world[inlier_matches[:, 0]]
    target_points = kpts2_world[inlier_matches[:, 1]]
    
    # Ensure arrays are 2D
    if source_points.ndim != 2:
        source_points = source_points.reshape(-1, 3)
    if target_points.ndim != 2:
        target_points = target_points.reshape(-1, 3)
    
    # Apply transformation
    source_pcd_temp = o3d.geometry.PointCloud()
    source_pcd_temp.points = o3d.utility.Vector3dVector(source_points)
    source_pcd_temp.transform(result.transformation)
    source_transformed = np.asarray(source_pcd_temp.points)
    
    rmse = np.sqrt(np.mean(np.sum((source_transformed - target_points)**2, axis=1)))
    
    return {
        'transformation': result.transformation,
        'num_matches': len(matches),
        'num_inliers': len(inlier_indices),
        'inlier_ratio': inlier_ratio,
        'rmse': rmse,
        'fitness': result.fitness
    }


def visualize_cross_camera_alignment(cloud_dir1, cloud_dir2, frame1_id, frame2_id,
                                    transformation, keypoint_dir1, keypoint_dir2):
    """
    Visualize alignment between two cameras.
    
    Args:
        cloud_dir1 (str): Directory with point clouds from camera 1
        cloud_dir2 (str): Directory with point clouds from camera 2
        frame1_id (str): Frame ID from camera 1
        frame2_id (str): Frame ID from camera 2
        transformation (np.ndarray): 4x4 transformation matrix
        keypoint_dir1 (str): Directory with keypoints from camera 1
        keypoint_dir2 (str): Directory with keypoints from camera 2
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
    
    # BEFORE: Both in their own camera coordinates
    pcd1_before = o3d.geometry.PointCloud(pcd1_cam)
    pcd2_before = o3d.geometry.PointCloud(pcd2_cam)
    pcd1_before.paint_uniform_color([1, 0, 0])  # Red - Camera 1
    pcd2_before.paint_uniform_color([0, 0, 1])  # Blue - Camera 2
    
    print(f"\n{'='*60}")
    print(f"BEFORE Cross-Camera Alignment")
    print(f"{'='*60}")
    print(f"Camera 1 frame {frame1_id} (red) and Camera 2 frame {frame2_id} (blue)")
    print("Different camera coordinate systems - not aligned")
    o3d.visualization.draw_geometries([pcd1_before, pcd2_before],
                                     window_name="BEFORE - Different Cameras")
    
    # AFTER: Transform camera 1 to align with camera 2
    alignment_transform = world_to_cam_2 @ transformation @ cam_to_world_1
    
    pcd1_after = o3d.geometry.PointCloud(pcd1_cam)
    pcd2_after = o3d.geometry.PointCloud(pcd2_cam)
    pcd1_after.transform(alignment_transform)
    pcd1_after.paint_uniform_color([0, 1, 0])  # Green - Aligned Camera 1
    pcd2_after.paint_uniform_color([0, 0, 1])  # Blue - Camera 2
    
    print(f"\n{'='*60}")
    print(f"AFTER Cross-Camera USIP Alignment")
    print(f"{'='*60}")
    print(f"Camera 1 aligned (green) with Camera 2 (blue)")
    print("Both cameras now in unified coordinate system")
    o3d.visualization.draw_geometries([pcd1_after, pcd2_after],
                                     window_name="AFTER - Collaborative SLAM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-camera alignment for collaborative SLAM")
    parser.add_argument("--video1_keypoints", type=str, required=True,
                       help="Keypoint directory for first video")
    parser.add_argument("--video2_keypoints", type=str, required=True,
                       help="Keypoint directory for second video")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for cross-camera matches")
    parser.add_argument("--overlap_threshold", type=float, default=0.3,
                       help="Minimum overlap ratio to consider frames as corresponding")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of best matches to return")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize best match")
    parser.add_argument("--cloud_dir1", type=str, default=None,
                       help="Point cloud directory for video 1 (for visualization)")
    parser.add_argument("--cloud_dir2", type=str, default=None,
                       help="Point cloud directory for video 2 (for visualization)")
    
    args = parser.parse_args()
    
    # Load keypoints from both videos
    print("Loading keypoints from both videos...")
    video1_kpts = load_all_keypoints(args.video1_keypoints)
    video2_kpts = load_all_keypoints(args.video2_keypoints)
    
    print(f"Video 1: {len(video1_kpts)} frames")
    print(f"Video 2: {len(video2_kpts)} frames")
    
    # Find cross-camera matches
    matches = find_cross_camera_matches(
        video1_kpts, video2_kpts,
        overlap_threshold=args.overlap_threshold,
        top_k=args.top_k
    )
    
    print(f"\nFound {len(matches)} frame pairs with overlap >= {args.overlap_threshold}")
    
    # Process each match to estimate transformation
    results = []
    
    for i, (frame1_id, frame2_id, overlap_score) in enumerate(matches):
        print(f"\n{'='*60}")
        print(f"Match {i+1}: Video1 frame {frame1_id} <-> Video2 frame {frame2_id}")
        print(f"Overlap score: {overlap_score:.2%}")
        
        # Get keypoints
        kpts1_world = video1_kpts[frame1_id]['keypoints_world']
        kpts2_world = video2_kpts[frame2_id]['keypoints_world']
        
        # Match keypoints
        kpt_matches = match_keypoints_cross_camera(kpts1_world, kpts2_world)
        print(f"Keypoint matches: {len(kpt_matches)}")
        
        if len(kpt_matches) >= 3:
            # Estimate transformation
            transform_result = estimate_camera_to_camera_transform(
                kpts1_world, kpts2_world, kpt_matches
            )
            
            if transform_result is not None:
                print(f"Inliers: {transform_result['num_inliers']}/{transform_result['num_matches']} "
                      f"({transform_result['inlier_ratio']:.2%})")
                print(f"RMSE: {transform_result['rmse']:.4f}m")
                
                results.append({
                    'video1_frame': frame1_id,
                    'video2_frame': frame2_id,
                    'overlap_score': float(overlap_score),
                    'transformation': transform_result['transformation'].tolist(),
                    'num_matches': int(transform_result['num_matches']),
                    'num_inliers': int(transform_result['num_inliers']),
                    'inlier_ratio': float(transform_result['inlier_ratio']),
                    'rmse': float(transform_result['rmse'])
                })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved {len(results)} cross-camera alignments to {args.output}")
    
    # Visualize best match if requested
    if args.visualize and len(results) > 0 and args.cloud_dir1 and args.cloud_dir2:
        best_match = results[0]
        print(f"\nVisualizing best match:")
        print(f"Video1 frame {best_match['video1_frame']} <-> Video2 frame {best_match['video2_frame']}")
        
        visualize_cross_camera_alignment(
            args.cloud_dir1, args.cloud_dir2,
            best_match['video1_frame'], best_match['video2_frame'],
            np.array(best_match['transformation']),
            args.video1_keypoints, args.video2_keypoints
        )
