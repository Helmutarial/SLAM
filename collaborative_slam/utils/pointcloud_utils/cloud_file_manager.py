"""
Cloud File Manager

Funciones para guardar, cargar y exportar nubes de puntos y poses.
"""
import open3d as o3d
import numpy as np
import os
import re
import json
import sys
from collaborative_slam.utils import file_utils
from collaborative_slam.views.open3d_visualization_classes import Open3DVisualization, update_camera_frame_from_vio, update_keyframes_from_mapping
import spectacularAI
import os
import sys
from collaborative_slam.utils.pointcloud_utils import save_pointclouds_and_poses

def save_pointclouds_and_poses(visu3D, cloud_points_folder, results_folder):
    """
    Save all point clouds from visu3D to .ply files and export camera poses to poses.json.
    Args:
        visu3D: Open3DVisualization instance containing point clouds and poses.
        cloud_points_folder: Folder to save .ply files.
        results_folder: Folder to save poses.json.
    """
    print(f"Saving point clouds to {cloud_points_folder}")
    poses_list = []
    for id in visu3D.pointClouds:
        pc = visu3D.pointClouds[id]
        filename = os.path.join(cloud_points_folder, f"{id}.ply")
        o3d.io.write_point_cloud(filename, pc.cloud)
        camToWorld = pc.camToWorld
        x = float(camToWorld[0, 3])
        y = float(camToWorld[1, 3])
        z = float(camToWorld[2, 3])
        poses_list.append({"frame": id, "x": x, "y": y, "z": z})
    poses_path = os.path.join(results_folder, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_list, f, indent=2)
    print(f"✅ Poses saved at: {poses_path}")

def extract_number(filename):
    """
    Extract number from filename for proper sorting.
    Args:
        filename (str): Filename to extract number from
    Returns:
        int: Extracted number or infinity if no number found
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_point_clouds(folder_path):
    """
    Load all point clouds from a folder.
    Args:
        folder_path (str): Path to folder containing .ply files
    Returns:
        tuple: (point_clouds, filenames)
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    files = sorted(files, key=extract_number)
    if not files:
        print(f"❌ No .ply files found in {folder_path}")
        return [], []
    point_clouds = []
    valid_files = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if pcd.has_points():
                point_clouds.append(pcd)
                valid_files.append(filename)
            else:
                print(f"⚠️ {filename} is empty, skipping...")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    return point_clouds, valid_files

def generate_pointclouds_from_data_folder(data_folder, results_folder):
    """
    Generates point clouds and camera poses from a data folder using spectacularAI and depthai.
    Saves .ply clouds and poses.json in the results folder structure.
    Replica exact logic from video_to_pointclouds.py, but fully programmatic.
    """
    voxelSize = 0
    cloud_points_folder = os.path.join(results_folder, 'cloud_points')
    os.makedirs(cloud_points_folder, exist_ok=True)
    visu3D = Open3DVisualization(voxelSize, False, False, False)
    def onVioOutput(vioOutput):
        update_camera_frame_from_vio(vioOutput, visu3D)
    def onMappingOutput(output):
        update_keyframes_from_mapping(output, visu3D)
    print(f"[INFO] Starting replay for {data_folder}")
    try:
        replay = spectacularAI.Replay(data_folder, onMappingOutput)
        replay.setOutputCallback(onVioOutput)
        replay.startReplay()
        visu3D.run()
        replay.close()
        save_pointclouds_and_poses(visu3D, cloud_points_folder, results_folder)
        print(f"[INFO] Point clouds and poses saved for {data_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to generate point clouds for {data_folder}: {e}")
        return
