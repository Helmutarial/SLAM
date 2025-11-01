"""
Visualize 3D point cloud of the environment in real-time, or playback your recordings and view their 3D point cloud.
Press 'H' to view Open3D point cloud viewer options.

Requirements: pip install open3d
"""

import spectacularAI
import depthai
import open3d as o3d
import numpy as np
import threading
import time
import os
import sys
import copy
import json
from collaborative_slam.utils import file_utils
from collaborative_slam.utils.pointcloud_utils import save_pointclouds_and_poses
from collaborative_slam.clases.SlamSceneManager import SlamSceneManager

# --- Funciones movidas del archivo eliminado ---
def update_camera_frame_from_vio(vioOutput, visu3D):
    """
    Update the camera frame in the visualization using VIO output.
    Args:
        vioOutput: VIO output object with camera pose information.
        visu3D: SlamSceneManager instance to update.
    """
    cameraPose = vioOutput.getCameraPose(0)
    camToWorld = cameraPose.getCameraToWorldMatrix()
    visu3D.updateCameraFrame(camToWorld)

def update_keyframes_from_mapping(output, visu3D):
    """
    Args:
        output: Mapping output object with updated keyframes.
        visu3D: SlamSceneManager instance to update.
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
    if hasattr(output, 'finalMap') and output.finalMap:
        print("Final map ready!")

def main():
    """
    Main function to run the point cloud visualization workflow step by step.
    """
    global visu3D
    # Select data folder interactively
    dataFolder = file_utils.select_data_folder()

    # Guardar todo en la carpeta de entrada seleccionada
    results_folder = dataFolder
    cloud_points_folder = os.path.join(results_folder, "cloud_points")
    os.makedirs(cloud_points_folder, exist_ok=True)
    voxelSize = 0
    visu3D = SlamSceneManager(voxelSize, False, False, False)

    print("Starting replay")
    replay = spectacularAI.Replay(dataFolder, onMappingOutput)
    replay.setOutputCallback(onVioOutput)
    replay.startReplay()
    visu3D.run()
    replay.close()

    save_pointclouds_and_poses(visu3D, cloud_points_folder, results_folder)



def onVioOutput(vioOutput):
    """
    Callback function for Visual-Inertial Odometry (VIO) output.
    Updates the camera frame in the visualization with the latest pose.
    Args:
        vioOutput: VIO output object containing camera pose information.
    """
    global visu3D
    update_camera_frame_from_vio(vioOutput, visu3D)

def onMappingOutput(output):
    """
    Callback function for SLAM mapping output.
    Updates the keyframes and point clouds in the visualization.
    Args:
        output: Mapping output object containing updated keyframes.
    """
    global visu3D
    update_keyframes_from_mapping(output, visu3D)

if __name__ == "__main__":
    main()
