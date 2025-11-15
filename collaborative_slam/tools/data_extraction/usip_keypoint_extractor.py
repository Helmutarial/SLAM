
import shutil
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import argparse
import os

def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for downsampling a point cloud.
    Args:
        points (np.ndarray): Nx3 array of points.
        num_samples (int): Number of points to sample.
    Returns:
        np.ndarray: Indices of sampled points.
    """
    N = points.shape[0]
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        sampled_indices[i] = farthest
        dist = np.sum((points - points[farthest]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return sampled_indices

def extract_keypoints_usip(points, num_keypoints=50, model=None, use_cuda=True):
    """
    Extract keypoints from a point cloud using USIP model or FPS fallback.
    Args:
        points (np.ndarray): Nx3 array of points.
        num_keypoints (int): Number of keypoints to extract.
        model: USIP model (optional, not used in fallback).
        use_cuda (bool): Use CUDA if model is provided.
    Returns:
        np.ndarray: Indices of keypoints.
        np.ndarray: Keypoint coordinates (K, 3).
    """
    # Fallback: FPS only (no USIP model loaded)
    print("Using Farthest Point Sampling (FPS) for keypoint selection")
    keypoint_indices = farthest_point_sampling(points, num_keypoints)
    keypoints = points[keypoint_indices]
    return keypoint_indices, keypoints


def select_video_folder():
    """
    Open a dialog to select a video data folder (e.g., VIDEO1).
    Returns:
        str: Path to the selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select video data folder (e.g., VIDEO1)")
    return folder_selected

def extract_and_save_keypoints(cloud_dir, keypoints_dir, num_keypoints=50):
    """
    Extract USIP keypoints for all .ply files in cloud_dir and save to keypoints_dir.
    """
    if os.path.isdir(keypoints_dir):
        print(f"Removing existing keypoints folder: {keypoints_dir}")
        shutil.rmtree(keypoints_dir)
    os.makedirs(keypoints_dir, exist_ok=True)
    ply_files = sorted([f for f in os.listdir(cloud_dir) if f.endswith('.ply')])
    if not ply_files:
        print(f"No .ply files found in {cloud_dir}")
        return []
    all_keypoints = []
    for ply_file in ply_files:
        cloud_path = os.path.join(cloud_dir, ply_file)
        pcd = o3d.io.read_point_cloud(cloud_path)
        points = np.asarray(pcd.points)
        # Extract keypoints (uses USIP model or FPS fallback)
        _, keypoints = extract_keypoints_usip(points, num_keypoints=num_keypoints)
        # Save keypoints as npz
        out_path = os.path.join(keypoints_dir, f"keypoints_{os.path.splitext(ply_file)[0]}.npz")
        np.savez(out_path, keypoints_camera=keypoints)
        all_keypoints.append(keypoints)
        print(f"Extracted {len(keypoints)} keypoints for {ply_file}")
    return all_keypoints

def visualize_keypoints_global(all_keypoints, point_size=3.0):
    """
    Visualize all keypoints merged in a single window.
    """
    merged_keypoints = np.vstack(all_keypoints)
    print(f"[INFO] Total number of keypoints: {merged_keypoints.shape[0]}")
    keypoints_pcd = o3d.geometry.PointCloud()
    keypoints_pcd.points = o3d.utility.Vector3dVector(merged_keypoints)
    keypoints_pcd.paint_uniform_color([1, 0, 0])  # Red
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="All keypoints (global)")
    vis.add_geometry(keypoints_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="USIP Keypoint Extractor & Visualizer")
    parser.add_argument('--video_folder', type=str, default=None, help='Path to video data folder (e.g., VIDEO1)')
    parser.add_argument('--num_keypoints', type=int, default=50, help='Number of keypoints per cloud')
    parser.add_argument('--point_size', type=float, default=3.0, help='Keypoint visualization size')
    args = parser.parse_args()

    if args.video_folder:
        video_folder = args.video_folder
    else:
        video_folder = select_video_folder()
    if not video_folder:
        print("No folder selected.")
        return
    cloud_dir = os.path.join(video_folder, "cloud_points")
    keypoints_dir = os.path.join(video_folder, "keypoints")
    if not os.path.isdir(cloud_dir):
        print(f"cloud_points/ folder not found in {video_folder}")
        return
    print(f"Extracting keypoints for all clouds in {cloud_dir}...")
    all_keypoints = extract_and_save_keypoints(cloud_dir, keypoints_dir, num_keypoints=args.num_keypoints)
    if not all_keypoints:
        print("No keypoints extracted.")
        return
    print("Visualizing global keypoints...")
    visualize_keypoints_global(all_keypoints, point_size=args.point_size)

if __name__ == "__main__":
    main()