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
from collaborative_slam.views.open3d_visualization_classes import update_camera_frame_from_vio, update_keyframes_from_mapping


def main():
    """
    Main function to run the point cloud visualization workflow step by step.
    """
    global visu3D
    # Select data folder interactively
    dataFolder = file_utils.select_data_folder()

    # Crear carpeta results y cloud_points en la raíz del workspace (TFM)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_folder = os.path.join(workspace_root, "results")
    cloud_points_folder = os.path.join(results_folder, "cloud_points")
    os.makedirs(results_folder, exist_ok=True)
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
