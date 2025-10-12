"""
Preprocessing & Filtering

Funciones para preprocesar, filtrar y limpiar nubes de puntos.
"""
import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsample the point cloud and extract FPFH features.
    Args:
        pcd: Open3D point cloud object.
        voxel_size: Voxel size for downsampling.
    Returns:
        Tuple (downsampled point cloud, FPFH features)
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh
