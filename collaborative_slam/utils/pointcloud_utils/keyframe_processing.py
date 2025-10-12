"""
Keyframe Processing

Funciones para convertir keyframes en nubes de puntos.
"""
import open3d as o3d

def process_keyframe_to_cloud(keyframe, voxel_size: float = 0.0, color_only: bool = False):
    """
    Convert a keyframe to Open3D point cloud.
    Args:
        keyframe: SpectacularAI keyframe
        voxel_size: Voxel size for downsampling (0 = no downsampling)
        color_only: Whether to filter points without color
    Returns:
        Open3D point cloud
    """
    cloud = o3d.geometry.PointCloud()
    points_camera = keyframe.pointCloud.getPositionData()
    points_camera *= 100
    cloud.points = o3d.utility.Vector3dVector(points_camera)
    if keyframe.pointCloud.hasColors():
        colors = keyframe.pointCloud.getRGB24Data() * 1./255
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    return cloud
