"""
Class for Open3D coordinate frame wrapper.
Allows updating pose and visualization.
"""

import open3d as o3d
import numpy as np

class CoordinateFrame:
    """
    Wrapper for Open3D coordinate frame, allows updating its pose.
    Args:
        scale: Scale of the coordinate frame.
    """
    def __init__(self, scale=0.25):
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(scale)
        self.camToWorld = np.identity(4)

    def updateWorldPose(self, camToWorld):
        """
        Update the world pose of the coordinate frame.
        Args:
            camToWorld: 4x4 transformation matrix.
        """
        prevWorldToCam = np.linalg.inv(self.camToWorld)
        prevToCurrent = np.matmul(camToWorld, prevWorldToCam)
        self.frame.transform(prevToCurrent)
        self.camToWorld = camToWorld
