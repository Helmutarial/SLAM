"""
Class for Open3D point cloud wrapper.
Allows updating pose and extracting cloud data.
"""

import open3d as o3d
import numpy as np
from .Status import Status

class PointCloud:
    """
    Wrapper for Open3D point clouds, allows updating their pose in the world.
    Args:
        keyFrame: KeyFrame object containing point cloud data.
        voxelSize: Voxel size for downsampling.
        colorOnly: If True, filter points without color.
        frame_id: Frame identifier (optional).
    """
    def __init__(self, keyFrame, voxelSize, colorOnly, frame_id=None):
        self.status = Status.NEW
        self.camToWorld = np.identity(4)
        self.cloud = self.__getKeyFramePointCloud(keyFrame, voxelSize, colorOnly)
        self.frame_id = frame_id

    def __getKeyFramePointCloud(self, keyFrame, voxelSize, colorOnly):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(keyFrame.pointCloud.getPositionData())
        if keyFrame.pointCloud.hasColors():
            colors = keyFrame.pointCloud.getRGB24Data() * 1./255
            cloud.colors = o3d.utility.Vector3dVector(colors)
        if keyFrame.pointCloud.hasNormals():
            cloud.normals = o3d.utility.Vector3dVector(keyFrame.pointCloud.getNormalData())
        if cloud.has_colors() and colorOnly:
            colors = np.asarray(cloud.colors)
            pointsWithColor = [i for i in range(len(colors)) if colors[i, :].any()]
            cloud = cloud.select_by_index(pointsWithColor)
        if voxelSize > 0:
            cloud = cloud.voxel_down_sample(voxelSize)
        return cloud

    def updateWorldPose(self, camToWorld):
        """
        Update the world pose of the point cloud.
        Args:
            camToWorld: 4x4 transformation matrix.
        """
        prevWorldToCam = np.linalg.inv(self.camToWorld)
        prevToCurrent = np.matmul(camToWorld, prevWorldToCam)
        self.cloud.transform(prevToCurrent)
        self.camToWorld = camToWorld
