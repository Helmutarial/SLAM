"""
Clases para visualización 3D con Open3D en SLAM.
Incluye: Status, PointCloud, CoordinateFrame, Open3DVisualization.
"""

import open3d as o3d
import numpy as np
from enum import Enum
import copy
import time

class Status(Enum):
    """
    Status of a point cloud or keyframe in the visualization.
    """
    VALID = 0
    NEW = 1
    UPDATED = 2
    REMOVED = 3

class PointCloud:
    """
    Wrapper for Open3D point clouds, allows updating their pose in the world.
    Args:
        keyFrame: KeyFrame object containing point cloud data.
        voxelSize: Voxel size for downsampling.
        colorOnly: If True, filter points without color.
    """
    def __init__(self, keyFrame, voxelSize, colorOnly):
        self.status = Status.NEW
        self.camToWorld = np.identity(4)
        self.cloud = self.__getKeyFramePointCloud(keyFrame, voxelSize, colorOnly)

    def __getKeyFramePointCloud(self, keyFrame, voxelSize, colorOnly):
        """
        Generate an Open3D point cloud from a keyFrame object.
        """
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

class Open3DVisualization:
    """
    Main class to manage the 3D visualization window, camera controls, and point cloud updates.
    Args:
        voxelSize: Voxel size for downsampling point clouds.
        cameraManual: If True, manual camera control.
        cameraSmooth: If True, smooth camera movement.
        colorOnly: If True, filter points without color.
    """
    def __init__(self, voxelSize, cameraManual, cameraSmooth, colorOnly):
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


def update_camera_frame_from_vio(vioOutput, visu3D):
    """
    Update the camera frame in the visualization using VIO output.
    Args:
        vioOutput: VIO output object with camera pose information.
        visu3D: Open3DVisualization instance to update.
    """
    cameraPose = vioOutput.getCameraPose(0)
    camToWorld = cameraPose.getCameraToWorldMatrix()
    visu3D.updateCameraFrame(camToWorld)

def update_keyframes_from_mapping(output, visu3D):
    """
    Update keyframes in the visualization using mapping output.
    Args:
        output: Mapping output object with updated keyframes.
        visu3D: Open3DVisualization instance to update.
    """
    for frameId in output.updatedKeyFrames:
        keyFrame = output.map.keyFrames.get(frameId)
        # Remove deleted key frames from visualisation
        if not keyFrame:
            if visu3D.containsKeyFrame(frameId): visu3D.removeKeyFrame(frameId)
            continue
        # Check that point cloud exists
        if not keyFrame.pointCloud: continue
        # Render key frame point clouds
        if visu3D.containsKeyFrame(frameId):
            # Existing key frame
            visu3D.updateKeyFrame(frameId, keyFrame)
        else:
            # New key frame
            visu3D.addKeyFrame(frameId, keyFrame)
    if output.finalMap:
        print("Final map ready!")