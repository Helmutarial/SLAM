from collaborative_slam.utils.pointcloud_utils.alignment import transform_trajectory

"""
USIP Collaborative SLAM Orchestration Script
--------------------------------------------
This script orchestrates the full USIP collaborative SLAM pipeline:
    1. Extract USIP keypoints for each camera
    2. Perform intra-camera keypoint matching
    3. Perform cross-camera keypoint matching and compute transformation
    4. Build collaborative map
    5. Visualize aligned point clouds and trajectories

Each step is modularized for clarity and maintainability.
"""

# Imports

import os
import sys
import subprocess
import json
import open3d as o3d
import numpy as np
from collaborative_slam.views.planview_visualization import visualize_planview
from collaborative_slam.utils.file_utils import select_data_folder


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(BASE_DIR, 'usip')

# Carpeta de resultados combinada en 'data' con los nombres de las dos carpetas de entrada
def get_combined_results_dir(data_dirs):
    data_base = os.path.dirname(data_dirs[0])
    name1 = os.path.basename(data_dirs[0])
    name2 = os.path.basename(data_dirs[1])
    combined_name = f"{name1}_AND_{name2}_results"
    results_dir = os.path.join(data_base, combined_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def run_subprocess(script, args):
    """
    Run a subprocess for a given script with arguments.
    Args:
        script (str): Path to the script to run.
        args (list): List of arguments for the script.
    """
    cmd = [sys.executable, script] + args
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def extract_usip_keypoints(data_dirs):
    """
    Extract USIP keypoints for each selected data folder.
    Args:
        data_dirs (list): List of data folder paths.
    """
    for data_dir in data_dirs:
        print(f"\n[Step 1] Extracting USIP keypoints for {os.path.basename(data_dir)}...")
        run_subprocess(
            os.path.join(TOOLS_DIR, 'batch_extract_keypoints.py'),
            ['--cloud_dir', os.path.join(data_dir, 'cloud_points'),
             '--output_dir', os.path.join(data_dir, 'keypoints')]
        )

def intra_camera_matching(data_dirs):
    """
    Perform intra-camera keypoint matching for each selected data folder.
    Args:
        data_dirs (list): List of data folder paths.
    """
    for data_dir in data_dirs:
        print(f"\n[Step 2] Intra-camera keypoint matching for {os.path.basename(data_dir)}...")
        keypoint_dir = os.path.join(data_dir, 'keypoints')
        output_path = os.path.join(data_dir, 'keypoint_matches.json')
        run_subprocess(
            os.path.join(TOOLS_DIR, 'keypoint_matcher.py'),
            ['--keypoint_dir', keypoint_dir, '--output', output_path]
        )

def cross_camera_alignment(data_dirs):
    """
    Perform cross-camera keypoint matching and compute transformation.
    Args:
        data_dirs (list): List of data folder paths (length 2 expected).
    """
    print(f"\n[Step 3] Cross-camera keypoint matching and transformation...")
    video1_keypoints_dir = os.path.join(data_dirs[0], 'keypoints')
    video2_keypoints_dir = os.path.join(data_dirs[1], 'keypoints')
    results_dir = get_combined_results_dir(data_dirs)
    output_path = os.path.join(results_dir, 'cross_camera_matches.json')
    run_subprocess(
        os.path.join(TOOLS_DIR, 'cross_camera_alignment.py'),
        [
            '--video1_keypoints', video1_keypoints_dir,
            '--video2_keypoints', video2_keypoints_dir,
            '--output', output_path
        ]
    )

def build_collaborative_map(data_dirs):
    """
    Build the collaborative map from aligned point clouds.
    Args:
        data_dirs (list): List of data folder paths (length 2 expected).
    """
    print(f"\n[Step 4] Building collaborative map...")
    results_dir = get_combined_results_dir(data_dirs)
    match_file = os.path.join(results_dir, 'cross_camera_matches.json')
    cloud_dir1 = os.path.join(data_dirs[0], 'cloud_points')
    cloud_dir2 = os.path.join(data_dirs[1], 'cloud_points')
    keypoint_dir1 = os.path.join(data_dirs[0], 'keypoints')
    keypoint_dir2 = os.path.join(data_dirs[1], 'keypoints')
    output_file = os.path.join(results_dir, 'collaborative_map_usip.ply')
    run_subprocess(
        os.path.join(TOOLS_DIR, 'collaborative_map_builder.py'),
        [
            '--match_file', match_file,
            '--cloud_dir1', cloud_dir1,
            '--cloud_dir2', cloud_dir2,
            '--keypoint_dir1', keypoint_dir1,
            '--keypoint_dir2', keypoint_dir2,
            '--output', output_file
        ]
    )

def load_poses_json(poses_path):
    """
    Load poses from a JSON file.
    Args:
        poses_path (str): Path to the poses.json file.
    Returns:
        np.ndarray: Array of positions (Nx2 or Nx3).
    """
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    # Try to support both [x, y, z] and dicts with keys
    if isinstance(poses[0], dict):
        if 'position' in poses[0]:
            return np.array([p['position'] for p in poses])
        elif 'x' in poses[0] and 'y' in poses[0]:
            return np.array([[p['x'], p['y']] for p in poses])
    return np.array(poses)

def load_point_cloud(pcd_path):
    """
    Load a point cloud from a .ply file.
    Args:
        pcd_path (str): Path to the .ply file.
    Returns:
        np.ndarray: Nx3 array of points.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def visualize_results(pcd_paths, pose_paths, transformed_pcd_path, transformed_pose_paths):
    """
    Visualize original and aligned point clouds and trajectories using shared planview function.
    """
    # Original
    pts1 = load_point_cloud(pcd_paths[0])
    pts2 = load_point_cloud(pcd_paths[1])
    traj1 = load_poses_json(pose_paths[0])
    traj2 = load_poses_json(pose_paths[1])
    visualize_planview(
        pts1, pts2, traj1, traj2,
        label1='Cloud 1', label2='Cloud 2',
        traj_color1='orange', traj_color2='blue',
        title='Original (USIP pipeline)'
    )
    # Aligned
    pts_aligned = load_point_cloud(transformed_pcd_path)
    traj1_aligned = load_poses_json(transformed_pose_paths[0])
    traj2_aligned = load_poses_json(transformed_pose_paths[1])
    visualize_planview(
        pts_aligned, np.zeros((0, 3)), traj1_aligned, traj2_aligned,
        label1='Collaborative Map', label2='',
        traj_color1='orange', traj_color2='blue',
        title='Aligned (USIP pipeline)'
    )


def main():
    """
    Main orchestration function for the USIP collaborative SLAM pipeline.
    """
    print("Selecciona la carpeta de datos de la primera cámara...")
    data_dir1 = select_data_folder()
    print("Selecciona la carpeta de datos de la segunda cámara...")
    data_dir2 = select_data_folder()
    data_dirs = [data_dir1, data_dir2]

    extract_usip_keypoints(data_dirs)
    intra_camera_matching(data_dirs)
    cross_camera_alignment(data_dirs)
    build_collaborative_map(data_dirs)

    print(f"\n[Step 5] Visualizing results...")
    import glob
    results_dir = get_combined_results_dir(data_dirs)
    def get_first_ply(data_dir):
        ply_files = glob.glob(os.path.join(data_dir, 'cloud_points', '*.ply'))
        if not ply_files:
            print(f"No se encontró ningún archivo .ply en {os.path.join(data_dir, 'cloud_points')}")
            return None
        return sorted(ply_files)[0]

    pcd_paths = [get_first_ply(data_dir) for data_dir in data_dirs]
    pose_paths = [
        os.path.join(data_dir, 'poses.json') for data_dir in data_dirs
    ]
    transformed_pcd_path = os.path.join(results_dir, 'collaborative_map_usip.ply')
    transformed_pose_paths = [
        os.path.join(results_dir, f'poses_{os.path.basename(data_dir)}_aligned.json') for data_dir in data_dirs
    ]
    # Transformar y guardar trayectorias alineadas usando la mejor transformación de cross_camera_matches.json
    import json

    # Cargar trayectorias originales
    traj1 = None
    traj2 = None
    try:
        with open(pose_paths[0], 'r') as f:
            poses1 = json.load(f)
        with open(pose_paths[1], 'r') as f:
            poses2 = json.load(f)
        # Soportar ambos formatos
        if isinstance(poses1[0], dict) and 'x' in poses1[0] and 'y' in poses1[0]:
            traj1 = np.array([[p['x'], p['y']] for p in poses1])
        else:
            traj1 = np.array(poses1)
        if isinstance(poses2[0], dict) and 'x' in poses2[0] and 'y' in poses2[0]:
            traj2 = np.array([[p['x'], p['y']] for p in poses2])
        else:
            traj2 = np.array(poses2)
    except Exception as e:
        print(f"No se pudieron cargar las trayectorias originales: {e}")

    # Cargar la transformación del primer par de matches realmente usado en la nube colaborativa
    best_T = None
    try:
        with open(os.path.join(results_dir, 'cross_camera_matches.json'), 'r') as f:
            matches = json.load(f)
        # Usar el primer match (el de menor RMSE, que es el primero tras el sort en build_collaborative_map)
        if matches and 'transformation' in matches[0]:
            best_T = np.array(matches[0]['transformation'])
            print(f"Usando la transformación del match: video1_frame={matches[0].get('video1_frame')}, video2_frame={matches[0].get('video2_frame')}")
    except Exception as e:
        print(f"No se pudo cargar la transformación de alineamiento: {e}")

    # Transformar y guardar trayectorias alineadas
    if best_T is not None and traj2 is not None:
        traj2_aligned = transform_trajectory(traj2, best_T)
        # Guardar trayectorias alineadas en la carpeta de resultados
        aligned_path2 = os.path.join(results_dir, f'poses_{os.path.basename(data_dirs[1])}_aligned.json')
        try:
            with open(aligned_path2, 'w') as f:
                json.dump(traj2_aligned.tolist(), f)
        except Exception as e:
            print(f"No se pudo guardar la trayectoria alineada: {e}")
    else:
        print("No se pudo transformar la trayectoria 2 (no hay transformación o trayectoria)")
    # La trayectoria 1 se deja igual (referencia)

    aligned_path1 = os.path.join(results_dir, f'poses_{os.path.basename(data_dirs[0])}_aligned.json')
    try:
        if traj1 is not None:
            with open(aligned_path1, 'w') as f:
                json.dump(traj1.tolist(), f)
        else:
            print("Trayectoria 1 no disponible, no se guarda archivo alineado para la referencia.")
    except Exception as e:
        print(f"No se pudo guardar la trayectoria 1: {e}")

    visualize_results(pcd_paths, pose_paths, transformed_pcd_path, transformed_pose_paths)

if __name__ == "__main__":
    main()
