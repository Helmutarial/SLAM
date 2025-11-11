"""
Batch Keypoint Extraction for Multiple Frames

This script processes all point cloud frames in a directory and extracts
keypoints using USIP, saving them for later use in alignment and matching.

Functions:
    - find_pointcloud_pairs: Find all PLY files and their corresponding JSON transforms
    - batch_extract_keypoints: Process multiple frames and save keypoints
    - save_keypoints_batch: Save all extracted keypoints to a single file
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent / "alignment_methods"))
from usip_keypoint_extractor import (
    load_usip_model,
    process_point_cloud_with_usip,
    save_keypoints
)


def find_pointcloud_pairs(cloud_dir):
    """
    Find all PLY files and their corresponding JSON transformation files.
    
    Args:
        cloud_dir (str): Directory containing PLY and JSON files
        
    Returns:
        list: List of tuples (ply_path, json_path, frame_id)
    """
    cloud_dir = Path(cloud_dir)
    pairs = []
    
    # Find all PLY files
    ply_files = sorted(cloud_dir.glob("*.ply"))
    
    for ply_file in ply_files:
        # Extract frame ID from filename (e.g., "1.ply" -> "1")
        frame_id = ply_file.stem
        
        # Look for corresponding JSON file
        json_file = cloud_dir / f"{frame_id}_camToWorld.json"
        
        if json_file.exists():
            pairs.append((str(ply_file), str(json_file), frame_id))
        else:
            print(f"Warning: No JSON transform found for {ply_file.name}, skipping...")
    
    return pairs


def batch_extract_keypoints(cloud_dir, output_dir, num_keypoints=512, 
                            model_path=None, use_cuda=True, save_individual=True):
    """
    Extract keypoints from all point clouds in a directory.
    
    Args:
        cloud_dir (str): Directory containing PLY and JSON files
        output_dir (str): Directory to save extracted keypoints
        num_keypoints (int): Number of keypoints to extract per frame
        model_path (str): Path to pre-trained USIP model (optional)
        use_cuda (bool): Whether to use GPU acceleration
        save_individual (bool): Whether to save keypoints for each frame individually
        
    Returns:
        dict: Dictionary mapping frame_id to extracted keypoints data
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all point cloud pairs
    pairs = find_pointcloud_pairs(cloud_dir)
    
    if len(pairs) == 0:
        print(f"No point cloud files found in {cloud_dir}")
        return {}
    
    print(f"Found {len(pairs)} point cloud frames")
    
    # Load USIP model once (reuse for all frames)
    model = None
    if model_path is not None:
        print(f"Loading USIP model from {model_path}...")
        model = load_usip_model(model_path=model_path, use_cuda=use_cuda, node_num=num_keypoints)
    
    # Process each frame
    all_keypoints = {}
    
    for ply_path, json_path, frame_id in tqdm(pairs, desc="Extracting keypoints"):
        try:
            # Extract keypoints
            result = process_point_cloud_with_usip(
                ply_file_path=ply_path,
                json_file_path=json_path,
                num_keypoints=num_keypoints,
                use_cuda=use_cuda,
                model=model
            )
            
            # Store results
            all_keypoints[frame_id] = {
                'keypoints_camera': result['keypoints_camera'],
                'keypoints_world': result['keypoints_world'],
                'keypoint_indices': result['keypoint_indices'],
                'cam_to_world': result['cam_to_world'],
                'ply_file': ply_path,
                'json_file': json_path
            }
            
            # Save individual frame keypoints if requested
            if save_individual:
                frame_output = output_dir / f"keypoints_{frame_id}.npz"
                np.savez(
                    frame_output,
                    keypoints_camera=result['keypoints_camera'],
                    keypoints_world=result['keypoints_world'],
                    keypoint_indices=result['keypoint_indices'],
                    cam_to_world=result['cam_to_world']
                )
        
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(all_keypoints)} frames")
    
    # Save summary
    summary_file = output_dir / "keypoints_summary.json"
    summary = {
        'num_frames': len(all_keypoints),
        'num_keypoints_per_frame': num_keypoints,
        'frames': list(all_keypoints.keys()),
        'model_path': model_path
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_file}")
    
    return all_keypoints


def save_keypoints_batch(all_keypoints, output_file):
    """
    Save all extracted keypoints to a single file.
    
    Args:
        all_keypoints (dict): Dictionary of keypoints data by frame_id
        output_file (str): Path to output file (.npz)
    """
    # Prepare data for saving
    save_dict = {}
    
    for frame_id, data in all_keypoints.items():
        save_dict[f'{frame_id}_camera'] = data['keypoints_camera']
        save_dict[f'{frame_id}_world'] = data['keypoints_world']
        save_dict[f'{frame_id}_indices'] = data['keypoint_indices']
        save_dict[f'{frame_id}_transform'] = data['cam_to_world']
    
    np.savez(output_file, **save_dict)
    print(f"Saved all keypoints to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract keypoints from point clouds")
    parser.add_argument("--cloud_dir", type=str, required=True, help="Directory containing PLY and JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted keypoints")
    parser.add_argument("--num_keypoints", type=int, default=512, help="Number of keypoints to extract per frame")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained USIP model")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no_individual", action="store_true", help="Don't save individual frame keypoints")
    parser.add_argument("--save_batch", type=str, default=None, help="Path to save all keypoints in a single file")
    
    args = parser.parse_args()
    
    # Extract keypoints
    all_keypoints = batch_extract_keypoints(
        cloud_dir=args.cloud_dir,
        output_dir=args.output_dir,
        num_keypoints=args.num_keypoints,
        model_path=args.model_path,
        use_cuda=not args.no_cuda,
        save_individual=not args.no_individual
    )
    
    # Save batch file if requested
    if args.save_batch is not None:
        save_keypoints_batch(all_keypoints, args.save_batch)
    
    print("\nBatch keypoint extraction complete!")
