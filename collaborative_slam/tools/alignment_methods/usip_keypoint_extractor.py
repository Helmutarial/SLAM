"""
USIP Keypoint Extractor

This module provides functionality to extract keypoints from 3D point clouds
using the USIP (Unsupervised Stable Interest Point) detector.

Functions:
    - load_point_cloud: Load point cloud from PLY file
    - extract_keypoints: Extract keypoints from a point cloud using USIP
    - transform_keypoints_to_world: Transform keypoints to world coordinates
    - process_point_cloud_with_usip: Complete pipeline to extract and transform keypoints
"""

import sys
import os
import json
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

# Add USIP to Python path
USIP_PATH = Path(__file__).parent.parent.parent / "external" / "USIP"
if str(USIP_PATH) not in sys.path:
    sys.path.insert(0, str(USIP_PATH))

# Import USIP modules
try:
    from models.keypoint_detector import ModelDetector
    from models import operations
    USIP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import USIP modules: {e}")
    USIP_AVAILABLE = False


def load_point_cloud(ply_file_path):
    """
    Load a point cloud from a PLY file.
    
    Args:
        ply_file_path (str): Path to the PLY file
        
    Returns:
        np.ndarray: Point cloud as numpy array (N, 3)
    """
    pcd = o3d.io.read_point_cloud(ply_file_path)
    points = np.asarray(pcd.points)
    return points


def load_camera_to_world_transform(json_file_path):
    """
    Load camera-to-world transformation matrix from JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing the 4x4 transformation matrix
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    with open(json_file_path, 'r') as f:
        transform = json.load(f)
    return np.array(transform)


class USIPOptions:
    """
    Simple options class for USIP model configuration.
    """
    def __init__(self, gpu_ids='0', node_num=512, scene='object'):
        self.gpu_ids = [int(x) for x in gpu_ids.split(',')]
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() and len(self.gpu_ids) >= 1 else "cpu")
        self.node_num = node_num
        self.scene = scene
        self.batch_size = 1
        self.input_pc_num = 5000
        self.surface_normal_len = 3
        self.activation = 'relu'
        self.normalization = 'batch'
        self.bn_momentum = 0.1
        self.bn_momentum_decay_step = None
        self.bn_momentum_decay = 0.6
        self.k = 1
        self.node_knn_k_1 = 32
        self.random_pc_dropout_lower_limit = 1.0
        self.rot_horizontal = False
        self.rot_3d = True
        self.rot_perturbation = False
        self.translation_perturbation = False
        self.loss_sigma_lower_bound = 0.0001
        self.keypoint_outlier_thre = 0.3
        self.keypoint_on_pc_alpha = 1.0
        self.keypoint_on_pc_type = 'point_to_point'
        self.lr = 0.001
        self.display_id = 1
        self.display_winsize = 256
        self.nThreads = 8


def load_usip_model(model_path=None, use_cuda=True, node_num=512):
    """
    Load a pre-trained USIP model.
    
    Args:
        model_path (str): Path to the pre-trained model checkpoint (optional)
        use_cuda (bool): Whether to use GPU acceleration
        node_num (int): Number of keypoints/nodes to detect
        
    Returns:
        ModelDetector: Loaded USIP model, or None if not available
    """
    if not USIP_AVAILABLE:
        print("Warning: USIP model not available, using fallback method")
        return None
    
    try:
        # Create options
        gpu_ids = '0' if use_cuda and torch.cuda.is_available() else '-1'
        opt = USIPOptions(gpu_ids=gpu_ids, node_num=node_num)
        
        # Create model
        model = ModelDetector(opt)
        
        # Load pre-trained weights if provided
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=opt.device)
            
            # Handle different checkpoint formats
            if 'detector' in checkpoint:
                model.detector.load_state_dict(checkpoint['detector'])
            elif 'state_dict' in checkpoint:
                model.detector.load_state_dict(checkpoint['state_dict'])
            else:
                model.detector.load_state_dict(checkpoint)
        else:
            print("Warning: No pre-trained model loaded, using random initialization")
        
        model.detector.eval()
        return model
    
    except Exception as e:
        print(f"Error loading USIP model: {e}")
        return None


def extract_keypoints_usip(point_cloud, model=None, num_keypoints=512, use_cuda=True):
    """
    Extract keypoints from a point cloud using USIP.
    
    Args:
        point_cloud (np.ndarray): Input point cloud (N, 3)
        model: Pre-trained USIP model (if None, a placeholder is used)
        num_keypoints (int): Number of keypoints to extract
        use_cuda (bool): Whether to use GPU acceleration
        
    Returns:
        np.ndarray: Keypoint indices (K,)
        np.ndarray: Keypoint coordinates (K, 3)
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # Normalize point cloud to unit sphere (USIP requirement)
    centroid = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / max_dist
    
    if model is None or not USIP_AVAILABLE:
        # Fallback: Use farthest point sampling
        print("Using Farthest Point Sampling (FPS) for keypoint selection")
        keypoint_indices = farthest_point_sampling(points_normalized, num_keypoints)
    else:
        # Use actual USIP model
        print("Using USIP model for keypoint detection")
        
        # Downsample or upsample to match expected input size
        target_size = model.opt.input_pc_num
        if point_cloud.shape[0] > target_size:
            # Downsample using FPS
            sample_indices = farthest_point_sampling(points_normalized, target_size)
            points_sampled = points_normalized[sample_indices]
        elif point_cloud.shape[0] < target_size:
            # Upsample by repeating points
            repeat_times = (target_size // point_cloud.shape[0]) + 1
            points_repeated = np.tile(points_normalized, (repeat_times, 1))
            points_sampled = points_repeated[:target_size]
            sample_indices = np.arange(target_size) % point_cloud.shape[0]
        else:
            points_sampled = points_normalized
            sample_indices = np.arange(point_cloud.shape[0])
        
        # Convert to tensor (1, 3, N)
        points_tensor = torch.from_numpy(points_sampled.T).float().unsqueeze(0).to(device)
        
        # Generate dummy surface normals (zeros for now)
        normals_tensor = torch.zeros_like(points_tensor).to(device)
        
        # Initialize node positions (uniform sampling or use FPS on sampled points)
        node_indices = farthest_point_sampling(points_sampled, num_keypoints)
        nodes_tensor = torch.from_numpy(points_sampled[node_indices].T).float().unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            node_recomputed, keypoints, sigmas, descriptors = model.forward(
                points_tensor, normals_tensor, nodes_tensor, is_train=False
            )
        
        # Extract keypoint coordinates (shape: 1, 3, K)
        keypoints_np = keypoints.squeeze(0).cpu().numpy().T  # (K, 3)
        
        # De-normalize keypoints
        keypoints_denorm = keypoints_np * max_dist + centroid
        
        # Find nearest neighbors in original point cloud to get indices
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(point_cloud)
        distances, keypoint_indices = nbrs.kneighbors(keypoints_denorm)
        keypoint_indices = keypoint_indices.flatten()
        
        # Use the actual keypoint coordinates from the model (already denormalized)
        keypoints = keypoints_denorm
        
        return keypoint_indices, keypoints
    
    # Get keypoint coordinates (in original scale) for FPS fallback
    keypoints = point_cloud[keypoint_indices]
    
    return keypoint_indices, keypoints


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for keypoint selection.
    
    Args:
        points (np.ndarray): Point cloud (N, 3)
        num_samples (int): Number of points to sample
        
    Returns:
        np.ndarray: Indices of sampled points
    """
    N = points.shape[0]
    if num_samples >= N:
        return np.arange(N)
    
    centroids = np.zeros(num_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids


def transform_keypoints_to_world(keypoints, cam_to_world_matrix):
    """
    Transform keypoints from camera coordinates to world coordinates.
    
    Args:
        keypoints (np.ndarray): Keypoints in camera coordinates (K, 3)
        cam_to_world_matrix (np.ndarray): 4x4 transformation matrix
        
    Returns:
        np.ndarray: Keypoints in world coordinates (K, 3)
    """
    # Convert to homogeneous coordinates
    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    
    # Apply transformation
    keypoints_world_homogeneous = (cam_to_world_matrix @ keypoints_homogeneous.T).T
    
    # Convert back to Cartesian coordinates
    keypoints_world = keypoints_world_homogeneous[:, :3]
    
    return keypoints_world


def process_point_cloud_with_usip(ply_file_path, json_file_path=None, num_keypoints=512, 
                                    use_cuda=True, model=None, model_path=None):
    """
    Complete pipeline to extract keypoints from a PLY file and transform to world coordinates.
    
    Args:
        ply_file_path (str): Path to the PLY file
        json_file_path (str): Path to the JSON file with camera-to-world transform (optional)
        num_keypoints (int): Number of keypoints to extract
        use_cuda (bool): Whether to use GPU acceleration
        model: Pre-trained USIP model (optional, will be loaded if None and model_path provided)
        model_path (str): Path to pre-trained USIP model checkpoint (optional)
        
    Returns:
        dict: Dictionary containing:
            - 'keypoint_indices': Indices of keypoints in the original point cloud
            - 'keypoints_camera': Keypoint coordinates in camera frame (K, 3)
            - 'keypoints_world': Keypoint coordinates in world frame (K, 3) if transform provided
            - 'cam_to_world': Transformation matrix (4, 4) if provided
    """
    # Load USIP model if not provided
    if model is None and model_path is not None:
        model = load_usip_model(model_path=model_path, use_cuda=use_cuda, node_num=num_keypoints)
    
    # Load point cloud
    point_cloud = load_point_cloud(ply_file_path)
    print(f"Loaded point cloud with {point_cloud.shape[0]} points from {ply_file_path}")
    
    # Extract keypoints
    keypoint_indices, keypoints_camera = extract_keypoints_usip(
        point_cloud, 
        model=model, 
        num_keypoints=num_keypoints, 
        use_cuda=use_cuda
    )
    print(f"Extracted {len(keypoint_indices)} keypoints")
    
    result = {
        'keypoint_indices': keypoint_indices,
        'keypoints_camera': keypoints_camera,
    }
    
    # Transform to world coordinates if transformation is provided
    if json_file_path is not None:
        cam_to_world = load_camera_to_world_transform(json_file_path)
        keypoints_world = transform_keypoints_to_world(keypoints_camera, cam_to_world)
        
        result['keypoints_world'] = keypoints_world
        result['cam_to_world'] = cam_to_world
        print(f"Transformed keypoints to world coordinates")
    
    return result


def save_keypoints(keypoints, output_path):
    """
    Save keypoints to a file.
    
    Args:
        keypoints (np.ndarray): Keypoints to save (K, 3)
        output_path (str): Path to output file (.npy or .txt)
    """
    if output_path.endswith('.npy'):
        np.save(output_path, keypoints)
    else:
        np.savetxt(output_path, keypoints)
    print(f"Saved keypoints to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract keypoints from point cloud using USIP")
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY file")
    parser.add_argument("--json", type=str, default=None, help="Path to camera-to-world JSON file")
    parser.add_argument("--num_keypoints", type=int, default=512, help="Number of keypoints to extract")
    parser.add_argument("--output", type=str, default=None, help="Path to save keypoints")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained USIP model")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Process point cloud
    result = process_point_cloud_with_usip(
        ply_file_path=args.ply,
        json_file_path=args.json,
        num_keypoints=args.num_keypoints,
        use_cuda=not args.no_cuda,
        model_path=args.model_path
    )
    
    # Save keypoints if output path is provided
    if args.output is not None:
        keypoints_to_save = result.get('keypoints_world', result['keypoints_camera'])
        save_keypoints(keypoints_to_save, args.output)
    
    print("\nKeypoint extraction complete!")
    print(f"Keypoints shape: {result['keypoints_camera'].shape}")
    if 'keypoints_world' in result:
        print(f"World keypoints shape: {result['keypoints_world'].shape}")
