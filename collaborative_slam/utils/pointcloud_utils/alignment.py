"""
Alignment & Registration

Funciones para alineación, ICP, correspondencia y registro de nubes.
"""
import open3d as o3d
from .preprocessing import preprocess_point_cloud

def align_clouds(source, target, voxel_size=5.0):
    """
    Align two point clouds using RANSAC + FPFH and return the initial transformation.
    Args:
        source: Source point cloud.
        target: Target point cloud.
        voxel_size: Voxel size for downsampling and feature extraction.
    Returns:
        Initial transformation matrix (4x4)
    """
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99)
    )
    return result_ransac.transformation

def compute_icp_rmse(source, target, initial_transform):
    """
    Run ICP after initial alignment and return RMSE error and refined transformation.
    Args:
        source: Source point cloud.
        target: Target point cloud.
        initial_transform: Initial transformation matrix (4x4).
    Returns:
        Tuple (RMSE error, refined transformation matrix)
    """
    import numpy as np
    threshold = 2.0
    source.transform(initial_transform)
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result.inlier_rmse, icp_result.transformation
