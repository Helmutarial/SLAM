"""
Keypoint Matching and Alignment

This script matches keypoints between consecutive frames and computes
relative transformations for point cloud alignment using USIP features.

Functions:
    - load_keypoints: Load extracted keypoints from .npz files
    - compute_features: Compute geometric features for keypoint matching
    - match_keypoints: Find correspondences between keypoint sets
    - estimate_transformation: Compute rigid transformation from matches
    - evaluate_alignment: Compute alignment metrics (RMSE, inlier ratio)
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import json
import argparse


def load_keypoints(keypoint_file):
    """
    Load keypoints from .npz file.
    
    Args:
        keypoint_file (str): Path to keypoints .npz file
        
    Returns:
        dict: Dictionary with keypoints_camera, keypoints_world, etc.
    """
    data = np.load(keypoint_file)
    return {
        'keypoints_camera': data['keypoints_camera'],
        'keypoints_world': data['keypoints_world'],
        'keypoint_indices': data['keypoint_indices'],
        'cam_to_world': data['cam_to_world']
    }


def compute_fpfh_features(pcd, search_radius=0.05):
    """
    Compute FPFH (Fast Point Feature Histogram) features for a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud
        search_radius (float): Search radius for feature computation
        
    Returns:
        o3d.pipelines.registration.Feature: FPFH features
    """
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
    )
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius * 5, max_nn=100)
    )
    
    return fpfh


def match_keypoints_fpfh(source_kpts, target_kpts, source_fpfh, target_fpfh, 
                         ratio_threshold=0.8):
    """
    Match keypoints using FPFH features and ratio test.
    
    Args:
        source_kpts (np.ndarray): Source keypoints (N, 3)
        target_kpts (np.ndarray): Target keypoints (M, 3)
        source_fpfh (o3d.pipelines.registration.Feature): Source FPFH features
        target_fpfh (o3d.pipelines.registration.Feature): Target FPFH features
        ratio_threshold (float): Lowe's ratio test threshold
        
    Returns:
        np.ndarray: Correspondence pairs (K, 2) with indices [source_idx, target_idx]
    """
    # Convert FPFH features to numpy arrays
    source_features = np.asarray(source_fpfh.data).T  # (N, 33)
    target_features = np.asarray(target_fpfh.data).T  # (M, 33)
    
    # Find two nearest neighbors for each source feature
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(target_features)
    distances, indices = nbrs.kneighbors(source_features)
    
    # Apply Lowe's ratio test
    matches = []
    for i in range(len(distances)):
        if distances[i][0] < ratio_threshold * distances[i][1]:
            matches.append([i, indices[i][0]])
    
    return np.array(matches)


def match_keypoints_geometry(source_kpts, target_kpts, max_distance=0.1):
    """
    Match keypoints using simple nearest neighbor in 3D space.
    
    Args:
        source_kpts (np.ndarray): Source keypoints (N, 3)
        target_kpts (np.ndarray): Target keypoints (M, 3)
        max_distance (float): Maximum distance for valid match
        
    Returns:
        np.ndarray: Correspondence pairs (K, 2) with indices [source_idx, target_idx]
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_kpts)
    distances, indices = nbrs.kneighbors(source_kpts)
    
    # Filter by distance threshold
    matches = []
    for i in range(len(distances)):
        if distances[i][0] < max_distance:
            matches.append([i, indices[i][0]])
    
    return np.array(matches)


def estimate_transformation_ransac(source_kpts, target_kpts, matches, 
                                   distance_threshold=0.05, max_iterations=1000):
    """
    Estimate rigid transformation using RANSAC.
    
    Args:
        source_kpts (np.ndarray): Source keypoints (N, 3)
        target_kpts (np.ndarray): Target keypoints (M, 3)
        matches (np.ndarray): Correspondence pairs (K, 2)
        distance_threshold (float): RANSAC inlier threshold
        max_iterations (int): Maximum RANSAC iterations
        
    Returns:
        dict: Transformation result with matrix, inliers, and metrics
    """
    if len(matches) < 3:
        return None
    
    # Create point clouds from matched keypoints
    source_matched = o3d.geometry.PointCloud()
    target_matched = o3d.geometry.PointCloud()
    
    source_matched.points = o3d.utility.Vector3dVector(source_kpts[matches[:, 0]])
    target_matched.points = o3d.utility.Vector3dVector(target_kpts[matches[:, 1]])
    
    # Create correspondence set
    corr = np.array([[i, i] for i in range(len(matches))])
    correspondence_set = o3d.utility.Vector2iVector(corr)
    
    # RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=source_matched,
        target=target_matched,
        corres=correspondence_set,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, 0.999)
    )
    
    # Compute metrics
    inlier_indices = np.asarray(result.correspondence_set)
    inlier_ratio = len(inlier_indices) / len(matches)
    
    # Get inlier matches
    inlier_matches = matches[inlier_indices]
    
    # Ensure inlier_matches is 2D array with shape (N, 2)
    if inlier_matches.ndim != 2 or inlier_matches.shape[1] != 2:
        # Fallback: use original matches for inliers
        inlier_matches = matches[inlier_indices].reshape(-1, 2)
    
    # Get corresponding source and target points
    source_points = source_kpts[inlier_matches[:, 0]]
    target_points = target_kpts[inlier_matches[:, 1]]
    
    # Ensure source_points and target_points are 2D arrays
    if source_points.ndim != 2:
        source_points = source_points.reshape(-1, 3)
    if target_points.ndim != 2:
        target_points = target_points.reshape(-1, 3)
    
    # Apply transformation using Open3D's transform method
    source_pcd_temp = o3d.geometry.PointCloud()
    source_pcd_temp.points = o3d.utility.Vector3dVector(source_points)
    source_pcd_temp.transform(result.transformation)
    source_transformed = np.asarray(source_pcd_temp.points)
    
    rmse = np.sqrt(np.mean(np.sum((source_transformed - target_points)**2, axis=1)))
    
    return {
        'transformation': result.transformation,
        'inliers': inlier_indices,
        'inlier_matches': inlier_matches,
        'num_matches': len(matches),
        'num_inliers': len(inlier_indices),
        'inlier_ratio': inlier_ratio,
        'rmse': rmse,
        'fitness': result.fitness
    }


def match_consecutive_frames(keypoint_dir, frame_pairs=None, use_fpfh=True, 
                            max_distance=0.1, ransac_threshold=0.05):
    """
    Match keypoints between consecutive frames.
    
    Args:
        keypoint_dir (str): Directory containing keypoint .npz files
        frame_pairs (list): Optional list of (frame1_id, frame2_id) tuples to match
        use_fpfh (bool): Whether to use FPFH features for matching
        max_distance (float): Maximum distance for geometric matching
        ransac_threshold (float): RANSAC inlier threshold
        
    Returns:
        dict: Dictionary of matching results by frame pair
    """
    keypoint_dir = Path(keypoint_dir)
    
    # Load summary to get frame list
    summary_file = keypoint_dir / "keypoints_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    frames = sorted(summary['frames'], key=lambda x: int(x))
    
    # Generate consecutive frame pairs if not provided
    if frame_pairs is None:
        frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    
    results = {}
    
    for frame1_id, frame2_id in frame_pairs:
        print(f"\nMatching frames {frame1_id} -> {frame2_id}")
        
        # Load keypoints
        kpts1_file = keypoint_dir / f"keypoints_{frame1_id}.npz"
        kpts2_file = keypoint_dir / f"keypoints_{frame2_id}.npz"
        
        kpts1_data = load_keypoints(kpts1_file)
        kpts2_data = load_keypoints(kpts2_file)
        
        # Use world coordinates for matching
        kpts1 = kpts1_data['keypoints_world']
        kpts2 = kpts2_data['keypoints_world']
        
        # Match keypoints
        if use_fpfh:
            # Create point clouds for FPFH computation
            pcd1 = o3d.geometry.PointCloud()
            pcd2 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(kpts1)
            pcd2.points = o3d.utility.Vector3dVector(kpts2)
            
            # Compute FPFH features
            fpfh1 = compute_fpfh_features(pcd1)
            fpfh2 = compute_fpfh_features(pcd2)
            
            # Match using FPFH
            matches = match_keypoints_fpfh(kpts1, kpts2, fpfh1, fpfh2)
        else:
            # Match using geometry only
            matches = match_keypoints_geometry(kpts1, kpts2, max_distance)
        
        print(f"  Found {len(matches)} initial matches")
        
        if len(matches) < 3:
            print("  Not enough matches for RANSAC")
            results[f"{frame1_id}_{frame2_id}"] = None
            continue
        
        # Estimate transformation with RANSAC
        transform_result = estimate_transformation_ransac(
            kpts1, kpts2, matches, 
            distance_threshold=ransac_threshold
        )
        
        if transform_result is not None:
            print(f"  Inliers: {transform_result['num_inliers']}/{transform_result['num_matches']} "
                  f"({transform_result['inlier_ratio']:.2%})")
            print(f"  RMSE: {transform_result['rmse']:.4f}m")
            
            results[f"{frame1_id}_{frame2_id}"] = transform_result
        else:
            results[f"{frame1_id}_{frame2_id}"] = None
    
    return results


def save_matching_results(results, output_file):
    """
    Save matching results to JSON file.
    
    Args:
        results (dict): Matching results dictionary
        output_file (str): Path to output JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for pair_id, result in results.items():
        if result is None:
            serializable_results[pair_id] = None
        else:
            serializable_results[pair_id] = {
                'transformation': result['transformation'].tolist(),
                'num_matches': int(result['num_matches']),
                'num_inliers': int(result['num_inliers']),
                'inlier_ratio': float(result['inlier_ratio']),
                'rmse': float(result['rmse']),
                'fitness': float(result['fitness'])
            }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nSaved matching results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match keypoints between frames")
    parser.add_argument("--keypoint_dir", type=str, required=True, 
                       help="Directory containing keypoint .npz files")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output JSON file for matching results")
    parser.add_argument("--use_fpfh", action="store_true", 
                       help="Use FPFH features for matching")
    parser.add_argument("--max_distance", type=float, default=0.1, 
                       help="Maximum distance for geometric matching")
    parser.add_argument("--ransac_threshold", type=float, default=0.05, 
                       help="RANSAC inlier threshold")
    parser.add_argument("--num_pairs", type=int, default=None, 
                       help="Number of consecutive pairs to process (for testing)")
    
    args = parser.parse_args()
    
    # Load summary to get frame list
    summary_file = Path(args.keypoint_dir) / "keypoints_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    frames = sorted(summary['frames'], key=lambda x: int(x))
    
    # Generate frame pairs
    frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    
    # Limit number of pairs if requested (for testing)
    if args.num_pairs is not None:
        frame_pairs = frame_pairs[:args.num_pairs]
        print(f"Processing first {args.num_pairs} frame pairs for testing")
    
    # Match keypoints
    results = match_consecutive_frames(
        keypoint_dir=args.keypoint_dir,
        frame_pairs=frame_pairs,
        use_fpfh=args.use_fpfh,
        max_distance=args.max_distance,
        ransac_threshold=args.ransac_threshold
    )
    
    # Save results
    save_matching_results(results, args.output)
    
    # Print summary statistics
    successful_matches = [r for r in results.values() if r is not None]
    
    if successful_matches:
        print(f"\n{'='*60}")
        print(f"MATCHING SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairs: {len(results)}")
        print(f"Successful matches: {len(successful_matches)}")
        print(f"Success rate: {len(successful_matches)/len(results):.2%}")
        
        avg_inlier_ratio = np.mean([r['inlier_ratio'] for r in successful_matches])
        avg_rmse = np.mean([r['rmse'] for r in successful_matches])
        
        print(f"\nAverage inlier ratio: {avg_inlier_ratio:.2%}")
        print(f"Average RMSE: {avg_rmse:.4f}m")
