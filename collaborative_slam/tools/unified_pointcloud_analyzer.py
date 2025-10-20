"""
Unified Point Cloud Analyzer

A comprehensive tool for analyzing and comparing 3D point clouds with multiple analysis modes.

This script unifies three different analysis approaches:
1. Quick comparison: Find best matches with minimum RMSE
2. Advanced correspondence: RANSAC + FPFH + ICP alignment  
3. Detailed analysis: Frame-by-frame RMSE evolution

Features:
- Multiple analysis modes in one tool
- Flexible configuration options
- Comprehensive output generation
- Optimized processing with skip options
- Interactive mode selection

Author: TFM Project
Version: 2.0 - Unified Analysis Tool
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from collaborative_slam.utils.file_utils import select_data_folder, prepare_results_folder


class UnifiedPointCloudAnalyzer:
    """
    Unified point cloud analysis tool with multiple analysis modes.
    """
    
    def __init__(self, config=None):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'icp_threshold': 3.0,           # ICP distance threshold
            'icp_max_iterations': 500,      # Maximum ICP iterations
            'voxel_size': 5.0,             # Voxel size for downsampling
            'ransac_threshold': 1.5,        # RANSAC distance threshold
            'skip_frames': 1,               # Process every Nth frame (1 = all frames)
            'output_format': 'png',         # Output image format
            'dpi': 150                      # Image resolution
        }
    
    def load_point_clouds(self, folder, sort_numerically=True):
        """
        Load all point clouds from a folder.
        
        Args:
            folder (str): Path to folder containing .ply files
            sort_numerically (bool): Sort by numeric value instead of alphabetically
            
        Returns:
            tuple: (point_clouds, filenames)
        """
        files = [f for f in os.listdir(folder) if f.endswith('.ply')]
        
        if sort_numerically:
            try:
                files.sort(key=lambda x: int(x.split('.')[0]))
            except (ValueError, IndexError):
                files.sort()  # Fallback to alphabetical
        else:
            files.sort()
        
        clouds = []
        valid_files = []
        
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                if pcd.has_points():
                    clouds.append(pcd)
                    valid_files.append(file)
                else:
                    print(f"⚠️ {file} is empty, skipping...")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
        
        return clouds, valid_files
    
    def apply_simple_icp(self, source, target):
        """
        Apply simple ICP registration.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            tuple: (fitness, rmse, transformation)
        """
        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, 
            self.config['icp_threshold'], 
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config['icp_max_iterations']
            )
        )
        
        return reg_result.fitness, reg_result.inlier_rmse, reg_result.transformation
    
    def preprocess_for_advanced_alignment(self, pcd, voxel_size):
        """
        Preprocess point cloud for advanced alignment (RANSAC + FPFH).
        
        Args:
            pcd: Point cloud to preprocess
            voxel_size (float): Voxel size for downsampling
            
        Returns:
            tuple: (downsampled_cloud, fpfh_features)
        """
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        
        # Compute FPFH features
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5, max_nn=100
            )
        )
        
        return pcd_down, pcd_fpfh
    
    def apply_advanced_alignment(self, source, target):
        """
        Apply advanced alignment with RANSAC + FPFH + ICP.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            tuple: (rmse, final_transformation)
        """
        voxel_size = self.config['voxel_size']
        
        # Preprocess both clouds
        source_down, source_fpfh = self.preprocess_for_advanced_alignment(source, voxel_size)
        target_down, target_fpfh = self.preprocess_for_advanced_alignment(target, voxel_size)
        
        distance_threshold = voxel_size * self.config['ransac_threshold']
        
        # RANSAC-based initial alignment
        try:
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, 
                mutual_filter=True,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99)
            )
            
            initial_transform = result_ransac.transformation
        except Exception as e:
            print(f"⚠️ RANSAC failed, using identity: {e}")
            initial_transform = np.eye(4)
        
        # Apply initial transformation
        source_temp = source.transform(initial_transform)
        
        # Refine with ICP
        try:
            icp_result = o3d.pipelines.registration.registration_icp(
                source_temp, target, 
                self.config['icp_threshold'], 
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            final_transform = np.dot(icp_result.transformation, initial_transform)
            return icp_result.inlier_rmse, final_transform
            
        except Exception as e:
            print(f"⚠️ ICP failed: {e}")
            return float('inf'), initial_transform
    
    def mode_quick_comparison(self, folder1, folder2, output_folder):
        """
        Mode 1: Quick comparison - find best matches with minimum RMSE.
        """
        print("🚀 Mode: Quick Comparison (Minimum RMSE)")
        print("="*50)
        
        clouds1, files1 = self.load_point_clouds(folder1)
        clouds2, files2 = self.load_point_clouds(folder2)
        
        if not clouds1 or not clouds2:
            print("❌ No valid point clouds found in one or both folders")
            return False
        
        min_rmse_values = []
        best_matches = []
        indices = []
        
        skip = self.config['skip_frames']
        
        for i, (cloud1, file1) in enumerate(zip(clouds1, files1)):
            if i % skip != 0:
                continue
                
            print(f"📊 Comparing {file1} against all clouds in folder2...")
            
            min_rmse = float('inf')
            best_idx = -1
            
            for j, (cloud2, file2) in enumerate(zip(clouds2, files2)):
                if j % skip != 0:
                    continue
                    
                _, rmse, _ = self.apply_simple_icp(cloud1, cloud2)
                
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_idx = j
            
            min_rmse_values.append(min_rmse)
            best_matches.append(best_idx)
            indices.append(i)
            
            if best_idx >= 0:
                print(f"✅ Best match: {files2[best_idx]} (RMSE: {min_rmse:.3f})")
        
        # Generate summary plot
        plt.figure(figsize=(12, 6))
        plt.plot(indices, min_rmse_values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel("Point Cloud Index (Folder 1)")
        plt.ylabel("Minimum RMSE")
        plt.title("Quick Comparison: Minimum RMSE for Each Point Cloud")
        plt.grid(True, alpha=0.3)
        
        output_plot = os.path.join(output_folder, "quick_comparison_min_rmse.png")
        plt.savefig(output_plot, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary plot saved: {output_plot}")
        return True
    
    def mode_advanced_correspondence(self, folder1, folder2, output_folder):
        """
        Mode 2: Advanced correspondence analysis with RANSAC + FPFH + ICP.
        """
        print("🎯 Mode: Advanced Correspondence Analysis")
        print("="*50)
        
        clouds1, files1 = self.load_point_clouds(folder1)
        clouds2, files2 = self.load_point_clouds(folder2)
        
        if not clouds1 or not clouds2:
            print("❌ No valid point clouds found in one or both folders")
            return False
        
        best_matches = []
        rmse_values = []
        skip = self.config['skip_frames']
        
        for i, (cloud1, file1) in enumerate(zip(clouds1, files1)):
            if i % skip != 0:
                continue
                
            print(f"🔍 Advanced analysis of {file1}...")
            
            best_rmse = float('inf')
            best_match = -1
            
            for j, (cloud2, file2) in enumerate(zip(clouds2, files2)):
                if j % skip != 0:
                    continue
                    
                rmse, _ = self.apply_advanced_alignment(cloud1, cloud2)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_match = j
            
            best_matches.append(best_match)
            rmse_values.append(best_rmse)
            
            if best_match >= 0:
                print(f"✅ Best correspondence: {files2[best_match]} (RMSE: {best_rmse:.3f})")
        
        # Generate correspondence plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(best_matches)), best_matches, marker='o', color='blue', alpha=0.7)
        plt.plot(range(len(best_matches)), best_matches, linestyle='--', color='gray', alpha=0.5)
        plt.xlabel("Point Cloud Index (Folder 1)")
        plt.ylabel("Best Match Index (Folder 2)")
        plt.title("Advanced Correspondence Mapping")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(rmse_values)), rmse_values, marker='o', color='red', linewidth=2)
        plt.xlabel("Point Cloud Index (Folder 1)")
        plt.ylabel("Best RMSE")
        plt.title("Correspondence Quality (RMSE)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_plot = os.path.join(output_folder, "advanced_correspondence_analysis.png")
        plt.savefig(output_plot, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"📊 Correspondence analysis saved: {output_plot}")
        return True
    
    def mode_detailed_analysis(self, folder1, folder2, output_folder):
        """
        Mode 3: Detailed frame-by-frame RMSE analysis.
        """
        print("📈 Mode: Detailed Frame-by-Frame Analysis")
        print("="*50)
        
        clouds1, files1 = self.load_point_clouds(folder1)
        clouds2, files2 = self.load_point_clouds(folder2)
        
        if not clouds1 or not clouds2:
            print("❌ No valid point clouds found in one or both folders")
            return False
        
        skip = self.config['skip_frames']
        generated_plots = 0
        
        for i, (cloud1, file1) in enumerate(zip(clouds1, files1)):
            if i % skip != 0:
                continue
                
            print(f"📊 Detailed analysis of {file1}...")
            
            rmse_values = []
            cloud2_indices = []
            
            for j, (cloud2, file2) in enumerate(zip(clouds2, files2)):
                if j % skip != 0:
                    continue
                    
                _, rmse, _ = self.apply_simple_icp(cloud1, cloud2)
                rmse_values.append(rmse)
                cloud2_indices.append(j)
            
            # Generate individual plot
            plt.figure(figsize=(10, 6))
            plt.plot(cloud2_indices, rmse_values, marker='o', linestyle='-', color='blue', linewidth=2)
            plt.xlabel("Point Cloud Index (Folder 2)")
            plt.ylabel("RMSE")
            plt.title(f"Detailed RMSE Analysis: {file1} vs All Clouds in Folder 2")
            plt.grid(True, alpha=0.3)
            
            # Highlight minimum RMSE point
            min_idx = np.argmin(rmse_values)
            plt.scatter(cloud2_indices[min_idx], rmse_values[min_idx], 
                       color='red', s=100, zorder=5, label=f'Min RMSE: {rmse_values[min_idx]:.3f}')
            plt.legend()
            
            output_plot = os.path.join(output_folder, f"detailed_rmse_{file1.replace('.ply', '')}.png")
            plt.savefig(output_plot, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            generated_plots += 1
        
        print(f"📊 Generated {generated_plots} detailed analysis plots")
        return True


def main():
    """Main function to run the unified point cloud analyzer."""
    parser = argparse.ArgumentParser(description="Unified Point Cloud Analyzer")
    parser.add_argument("--mode", type=str, choices=['quick', 'advanced', 'detailed', 'all'], 
                       default='all', help="Analysis mode")
    parser.add_argument("--folder1", type=str, default=None,
                       help="First point cloud folder for comparison")
    parser.add_argument("--folder2", type=str, default=None,
                       help="Second point cloud folder for comparison")
    parser.add_argument("--skip", type=int, default=1, 
                       help="Process every Nth frame (1 = all frames)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output folder (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("🔍 UNIFIED POINT CLOUD ANALYZER")
    print("="*60)
    print("🎯 Analysis Modes Available:")
    print("   • Quick: Find best matches with minimum RMSE")
    print("   • Advanced: RANSAC + FPFH + ICP correspondence analysis")
    print("   • Detailed: Frame-by-frame RMSE evolution graphs")
    print("   • All: Run all three modes")
    print()
    
    # Get input folders from args or user selection
    if args.folder1 and args.folder2:
        folder1 = args.folder1
        folder2 = args.folder2
        print(f"📁 Using provided folders:")
        print(f"   Folder 1: {folder1}")
        print(f"   Folder 2: {folder2}")
        
        # Validate folders exist
        if not os.path.exists(folder1):
            print(f"❌ Folder 1 not found: {folder1}")
            return 1
        if not os.path.exists(folder2):
            print(f"❌ Folder 2 not found: {folder2}")
            return 1
    else:
        # Interactive folder selection
        print("📁 Select FIRST point cloud folder:")
        folder1 = select_data_folder()
        if not folder1:
            print("❌ No first folder selected")
            return 1
        
        print("📁 Select SECOND point cloud folder:")
        folder2 = select_data_folder()
        if not folder2:
            print("❌ No second folder selected")
            return 1
    
    # Prepare output folder
    if args.output:
        output_folder = args.output
        os.makedirs(output_folder, exist_ok=True)
    else:
        folder1_name = os.path.basename(folder1)
        results_base = prepare_results_folder(folder1)
        output_folder = os.path.join(os.path.dirname(results_base), "pointcloud_analysis")
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"📂 Input folder 1: {os.path.basename(folder1)}")
    print(f"📂 Input folder 2: {os.path.basename(folder2)}")
    print(f"💾 Output folder: {output_folder}")
    print(f"⚙️ Skip frames: {args.skip}")
    print(f"🔧 Analysis mode: {args.mode}")
    print()
    
    # Initialize analyzer
    config = {
        'skip_frames': args.skip,
        'icp_threshold': 3.0,
        'icp_max_iterations': 500,
        'voxel_size': 5.0,
        'ransac_threshold': 1.5,
        'output_format': 'png',
        'dpi': 150
    }
    
    analyzer = UnifiedPointCloudAnalyzer(config)
    
    try:
        success_count = 0
        
        if args.mode in ['quick', 'all']:
            if analyzer.mode_quick_comparison(folder1, folder2, output_folder):
                success_count += 1
            print()
        
        if args.mode in ['advanced', 'all']:
            if analyzer.mode_advanced_correspondence(folder1, folder2, output_folder):
                success_count += 1
            print()
        
        if args.mode in ['detailed', 'all']:
            if analyzer.mode_detailed_analysis(folder1, folder2, output_folder):
                success_count += 1
            print()
        
        # Final summary
        print("="*60)
        print("✅ ANALYSIS COMPLETED")
        print(f"📊 Successful analyses: {success_count}")
        print(f"💾 Results saved to: {output_folder}")
        print("="*60)
        
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)