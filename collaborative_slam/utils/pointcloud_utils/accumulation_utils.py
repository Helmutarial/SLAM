"""
Point Cloud Utilities

This module provides utility functions for point cloud processing and manipulation.

Functions:
- accumulate_point_clouds: Accumulate and merge multiple point clouds from recorded data
- process_keyframe_to_cloud: Convert a keyframe to Open3D point cloud
- merge_point_clouds: Merge multiple point clouds into one
- save_accumulated_clouds: Save accumulated point clouds to files
"""

import spectacularAI
import open3d as o3d
import numpy as np
import os
from typing import List, Optional, Tuple


def process_keyframe_to_cloud(keyframe, voxel_size: float = 0.0, color_only: bool = False) -> o3d.geometry.PointCloud:
    """
    Convert a keyframe to Open3D point cloud.
    
    Args:
        keyframe: SpectacularAI keyframe
        voxel_size: Voxel size for downsampling (0 = no downsampling)
        color_only: Whether to filter points without color
        
    Returns:
        Open3D point cloud
    """
    cloud = o3d.geometry.PointCloud()
    
    # Get points in camera coordinate system
    points_camera = keyframe.pointCloud.getPositionData()
    
    # Convert to centimeters
    points_camera *= 100
    
    cloud.points = o3d.utility.Vector3dVector(points_camera)

    # Add colors if available
    if keyframe.pointCloud.hasColors():
        colors = keyframe.pointCloud.getRGB24Data() * 1./255
        cloud.colors = o3d.utility.Vector3dVector(colors)

    # Apply voxel downsampling if specified
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)

    return cloud


def merge_point_clouds(clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """
    Merge multiple point clouds into one.
    
    Args:
        clouds: List of Open3D point clouds
        
    Returns:
        Merged point cloud
    """
    merged_cloud = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged_cloud += cloud
    return merged_cloud


def save_accumulated_clouds(accumulated_clouds: List[o3d.geometry.PointCloud], 
                          output_folder: str, 
                          frame_counter: int,
                          camera_matrix: Optional[np.ndarray] = None) -> str:
    """
    Save accumulated point clouds to a PLY file.
    
    Args:
        accumulated_clouds: List of point clouds to merge and save
        output_folder: Output directory
        frame_counter: Frame number for filename
        camera_matrix: Camera transformation matrix (optional)
        
    Returns:
        Path to saved file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge all accumulated clouds
    merged_cloud = merge_point_clouds(accumulated_clouds)
    
    # Transform to camera coordinate system if matrix provided
    if camera_matrix is not None:
        merged_cloud.transform(camera_matrix)

    # Save to PLY file
    ply_filename = os.path.join(output_folder, f"{frame_counter:07d}.ply")
    o3d.io.write_point_cloud(ply_filename, merged_cloud)
    
    return ply_filename


def accumulate_point_clouds(data_folder: str, 
                          output_folder: str = "accumulated_clouds",
                          accumulation_count: int = 5,
                          voxel_size: float = 0.0,
                          color_only: bool = False,
                          verbose: bool = True) -> List[str]:
    """
    Process recorded data and accumulate point clouds in batches.
    
    Args:
        data_folder: Folder containing recorded SpectacularAI session
        output_folder: Output folder for accumulated point clouds
        accumulation_count: Number of clouds to accumulate before saving
        voxel_size: Voxel size for downsampling (0 = no downsampling)
        color_only: Filter points without color
        verbose: Print progress messages
        
    Returns:
        List of saved file paths
    """
    if verbose:
        print(f"🔄 Starting point cloud accumulation...")
        print(f"   📁 Input: {data_folder}")
        print(f"   💾 Output: {output_folder}")
        print(f"   📊 Accumulation: {accumulation_count} clouds per file")
        print(f"   🔧 Voxel size: {voxel_size}m")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize variables
    accumulated_clouds = []
    frame_counter = 0
    current_camera_matrix = None
    saved_files = []

    def save_current_batch():
        nonlocal accumulated_clouds, frame_counter, current_camera_matrix, saved_files
        
        if accumulated_clouds:
            saved_file = save_accumulated_clouds(
                accumulated_clouds, 
                output_folder, 
                frame_counter, 
                current_camera_matrix
            )
            saved_files.append(saved_file)
            
            if verbose:
                print(f"✅ Saved accumulated cloud: {os.path.basename(saved_file)} ({len(accumulated_clouds)} clouds merged)")
            
            frame_counter += 1
            accumulated_clouds = []

    def on_mapping_output(output):
        nonlocal accumulated_clouds, current_camera_matrix
        
        for frame_id in output.updatedKeyFrames:
            keyframe = output.map.keyFrames.get(frame_id)
            if not keyframe:
                continue

            # Get camera transformation matrix
            cam_to_world = keyframe.frameSet.primaryFrame.cameraPose.getCameraToWorldMatrix()
            current_camera_matrix = np.linalg.inv(cam_to_world)

            # Process keyframe to point cloud
            cloud = process_keyframe_to_cloud(keyframe, voxel_size, color_only)
            
            # Transform to global coordinates
            transformed_cloud = cloud.transform(cam_to_world)
            
            # Accumulate the cloud
            accumulated_clouds.append(transformed_cloud)

            # Save batch if we've accumulated enough clouds
            if len(accumulated_clouds) >= accumulation_count:
                save_current_batch()

        # Save final batch if this is the final map
        if output.finalMap:
            save_current_batch()
            if verbose:
                print(f"🏁 Final map processed!")

    try:
        if verbose:
            print("🎬 Starting replay...")
        
        # Start SpectacularAI replay
        replay = spectacularAI.Replay(data_folder, on_mapping_output)
        replay.startReplay()
        replay.close()
        
        # Save any remaining clouds
        save_current_batch()
        
        if verbose:
            print(f"✅ Point cloud accumulation completed!")
            print(f"📊 Generated {len(saved_files)} accumulated point cloud files")
            
        return saved_files
        
    except Exception as e:
        print(f"❌ Error during point cloud accumulation: {e}")
        return saved_files


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Accumulate point clouds from recorded data")
    parser.add_argument("--dataFolder", required=True, 
                       help="Folder containing the recorded session")
    parser.add_argument("--outputFolder", default="accumulated_clouds",
                       help="Output folder for accumulated point clouds")
    parser.add_argument("--accumulation", type=int, default=5,
                       help="Number of clouds to accumulate before saving")
    parser.add_argument("--voxel", type=float, default=0.0,
                       help="Voxel size (m) for downsampling")
    parser.add_argument("--color", action="store_true",
                       help="Filter points without color")
    
    args = parser.parse_args()
    
    print("📊 POINT CLOUD ACCUMULATOR")
    print("="*50)
    
    saved_files = accumulate_point_clouds(
        data_folder=args.dataFolder,
        output_folder=args.outputFolder,
        accumulation_count=args.accumulation,
        voxel_size=args.voxel,
        color_only=args.color,
        verbose=True
    )
    
    if saved_files:
        print(f"\n🎯 Successfully generated {len(saved_files)} accumulated point cloud files:")
        for file_path in saved_files:
            print(f"   • {file_path}")
    else:
        print("\n❌ No point cloud files were generated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)