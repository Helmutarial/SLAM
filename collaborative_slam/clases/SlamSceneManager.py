import open3d as o3d
import numpy as np
import copy
import time
from enum import Enum
from .CoordinateFrame import CoordinateFrame
from .PointCloud import PointCloud
from .Status import Status

class SlamSceneManager:
    """
    Main class to manage the SLAM scene: keyframes, point clouds and optional visualization.
    Args:
        voxelSize: Voxel size for downsampling point clouds.
        cameraManual: If True, manual camera control.
        cameraSmooth: If True, smooth camera movement.
        colorOnly: If True, filter points without color.
        enableVisualization: If True, crea la ventana de Open3D. Si False, solo gestiona datos.
    """
    def __init__(self, voxelSize, cameraManual, cameraSmooth, colorOnly, enableVisualization=True):
        self.shouldClose = False
        self.cameraFrame = CoordinateFrame()
        self.pointClouds = {}
        self.pointCloudsNoWolrld = {}
        self.voxelSize = voxelSize
        self.cameraFollow = not cameraManual
        self.cameraSmooth = cameraSmooth
        self.colorOnly = colorOnly
        self.prevPos = None
        self.prevCamPos = None
        self.vis = None
        self.viewControl = None
        if enableVisualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(self.cameraFrame.frame, reset_bounding_box=False)
            self.viewControl = self.vis.get_view_control()
            renderOption = self.vis.get_render_option()
            renderOption.point_size = 2
            renderOption.light_on = False

    def run(self):
        """
        Main loop for the Open3D visualization window. Updates geometries and handles window events.
        """
        print("Close the window to stop mapping")
        while not self.shouldClose:
            self.shouldClose = not self.vis.poll_events()
            self.vis.update_geometry(self.cameraFrame.frame)
            for pcId in list(self.pointClouds.keys()):
                pc = self.pointClouds[pcId]
                if pc.status == Status.VALID:
                    continue
                elif pc.status == Status.NEW:
                    reset = len(self.pointClouds) == 1
                    self.vis.add_geometry(pc.cloud, reset_bounding_box=reset)
                    pc.status = Status.VALID
                elif pc.status == Status.UPDATED:
                    self.vis.update_geometry(pc.cloud)
                    pc.status = Status.VALID
                elif pc.status == Status.REMOVED:
                    self.vis.remove_geometry(pc.cloud, reset_bounding_box=False)
                    del self.pointClouds[pcId]
            self.vis.update_renderer()
            time.sleep(0.01)
        self.vis.destroy_window()

    def updateCameraFrame(self, camToWorld):
        """
        Update the camera frame pose and camera view in the visualization.
        Args:
            camToWorld: 4x4 transformation matrix.
        """
        self.cameraFrame.updateWorldPose(camToWorld)
        if self.cameraFollow:
            pos = camToWorld[0:3, 3]
            forward = camToWorld[0:3, 2]
            upVector = np.array([0, 0, 1])
            camPos = pos - forward * 0.1 + upVector * 0.05
            if self.cameraSmooth and self.prevPos is not None:
                alpha = np.array([0.01, 0.01, 0.001])
                camPos = camPos * alpha + self.prevCamPos * (np.array([1, 1, 1])  - alpha)
                pos = pos * alpha + self.prevPos * (np.array([1, 1, 1]) - alpha)
            self.prevPos = pos
            self.prevCamPos = camPos
            viewDir = pos - camPos
            viewDir /= np.linalg.norm(viewDir)
            leftDir = np.cross(upVector, viewDir)
            upDir = np.cross(viewDir, leftDir)
            self.viewControl.set_lookat(pos)
            self.viewControl.set_front(-viewDir)
            self.viewControl.set_up(upDir)
            self.viewControl.set_zoom(0.3)

    def containsKeyFrame(self, keyFrameId):
        """
        Check if a keyframe exists in the visualization.
        Args:
            keyFrameId: Identifier of the keyframe.
        Returns:
            True if exists, False otherwise.
        """
        return keyFrameId in self.pointClouds

    def addKeyFrame(self, keyFrameId, keyFrame):
        """
        Add a new keyframe and its point cloud to the visualization.
        Args:
            keyFrameId: Identifier of the keyframe.
            keyFrame: KeyFrame object.
        """
        camToWorld = keyFrame.frameSet.primaryFrame.cameraPose.getCameraToWorldMatrix()
        pc = PointCloud(keyFrame, self.voxelSize, self.colorOnly)
        self.pointCloudsNoWolrld[keyFrameId] = copy.deepcopy(pc)
        pc.updateWorldPose(camToWorld)
        self.pointClouds[keyFrameId] = pc

    def updateKeyFrame(self, keyFrameId, keyFrame):
        """
        Update an existing keyframe's pose and point cloud.
        Args:
            keyFrameId: Identifier of the keyframe.
            keyFrame: KeyFrame object.
        """
        camToWorld = keyFrame.frameSet.primaryFrame.cameraPose.getCameraToWorldMatrix()
        pc = self.pointClouds[keyFrameId]
        pc.updateWorldPose(camToWorld)
        pc.status = Status.UPDATED

    def removeKeyFrame(self, keyFrameId):
        """
        Remove a keyframe and its point cloud from the visualization.
        Args:
            keyFrameId: Identifier of the keyframe.
        """
        pc = self.pointClouds[keyFrameId]
        pc.status = Status.REMOVED

