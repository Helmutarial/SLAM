"""
Cloud File Manager

Funciones para guardar, cargar y exportar nubes de puntos y poses.
"""
import open3d as o3d
import numpy as np
import os
import re
import json

def save_pointclouds_and_poses(visu3D, cloud_points_folder, results_folder):
    print(f"Saving point clouds to {cloud_points_folder}")
    # Buscar data.jsonl en la carpeta de datos
    data_jsonl_path = None
    for root, dirs, files in os.walk(results_folder):
        for d in dirs:
            candidate = os.path.join(root, d, "data.jsonl")
            if os.path.exists(candidate):
                data_jsonl_path = candidate
                break
        if data_jsonl_path:
            break
    timestamps_by_frame = {}
    if data_jsonl_path:
        with open(data_jsonl_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "frames" in entry and "number" in entry:
                        frame_num = entry["number"]
                        timestamps_by_frame[frame_num] = entry.get("time", None)
                except Exception:
                    continue
    poses_list = []
    for id in visu3D.pointClouds:
        pc = visu3D.pointClouds[id]
        filename = os.path.join(cloud_points_folder, f"{id}.ply")
        o3d.io.write_point_cloud(filename, pc.cloud)
        camToWorld = pc.camToWorld
        # Guardar la matriz camToWorld en JSON
        camToWorld_path = os.path.join(cloud_points_folder, f"{id}_camToWorld.json")
        with open(camToWorld_path, "w") as f:
            json.dump(camToWorld.tolist(), f, indent=2)
        # Eliminado guardado de .npy
        x = float(camToWorld[0, 3])
        y = float(camToWorld[1, 3])
        z = float(camToWorld[2, 3])
        # Buscar el timestamp exacto o el más cercano anterior en data.jsonl
        timestamp = timestamps_by_frame.get(id, None)
        if timestamp is None and len(timestamps_by_frame) > 0:
            menores = [f for f in timestamps_by_frame if f < id]
            if menores:
                closest = max(menores)
                timestamp = timestamps_by_frame[closest]
            else:
                mayores = [f for f in timestamps_by_frame if f > id]
                if mayores:
                    closest = min(mayores)
                    timestamp = timestamps_by_frame[closest]
        # Usar el frame_id original si existe
        # Asegurar que frame nunca sea None/null
        frame_id = getattr(pc, 'frame_id', id)
        if frame_id is None:
            frame_id = id if id is not None else -1
        pose_dict = {"frame": frame_id, "x": x, "y": y, "z": z}
        if timestamp is not None:
            pose_dict["timestamp"] = timestamp
        poses_list.append(pose_dict)
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
