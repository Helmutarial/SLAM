"""
Conservative Trajectory Extractor with Sensor Fusion

This is the main script for extracting camera trajectories from visual and sensor data.
It uses conservative filtering to distinguish between camera rotation and actual movement,
leveraging accelerometer and gyroscope data for intelligent movement classification.

Features:
- Frame filtering based on feature count
- Sensor-based movement classification (still/rotation/translation)
- Gyroscope-directed movement calculation
- Temporal synchronization with video
- Automatic video playback synchronization

Author: TFM Project
Version: 2.0 - Modular and Clean
"""

import os
import sys
import glob
from collaborative_slam.utils.trayectory_utils import data_processing, trajectory_filtering, sensor_processing        
from utils.file_utils import select_data_folder
from collaborative_slam.views.visualize_trajectory import visualize_poses_directly
from views.video_player import find_video_file, play_video_synchronized, get_video_duration


class TrajectoryExtractor:
    """
    Main class for trajectory extraction with sensor fusion.
    """
    
    def __init__(self, config=None):
        """
        Initialize the trajectory extractor.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or self._default_config()
        self.data = {}
        self.results = {}
    
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'min_features': 10,
            'sensor_window_ms': 200,
            'filtering': {
                'movement_threshold_high': 25,
                'movement_threshold_uncertain': 40,
                'movement_scale_confirmed': 0.2,
                'movement_scale_uncertain': 0.05,
                'max_movement_per_frame': 15,
                'early_frame_limit': 10,
                'early_frame_threshold': 100
            },
            'visualization': {
                'fps': 25,
                'auto_play_video': True,
                'video_sync_buffer': 0.5
            }
        }
    
    def load_data(self, jsonl_path):
        """
        Load and parse data from JSONL file.
        
        Args:
            jsonl_path (str): Path to the JSONL data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔍 Loading and analyzing data...")
            
            # Load raw entries
            entries = data_processing.load_jsonl_data(jsonl_path)
            
            # Extract visual poses and frame info
            raw_poses, frame_info = trajectory_filtering.filter_low_feature_frames(
                entries, 
                self.config['min_features']
            )
            
            # Extract sensor data
            sensor_data = data_processing.extract_sensor_data(entries)
            
            # Store data
            self.data = {
                'raw_poses': raw_poses,
                'frame_info': frame_info,
                'sensor_data': sensor_data,
                'entries': entries
            }
            
            # Print summary and validate
            data_processing.print_data_summary(raw_poses, frame_info, sensor_data)
            validation = data_processing.validate_data_integrity(raw_poses, frame_info, sensor_data)
            
            return validation['is_valid']
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def process_trajectory(self):
        """
        Process trajectory data using sensor-based filtering.
        
        Returns:
            list: Processed trajectory poses
        """
        if not self.data:
            print("❌ No data loaded. Call load_data() first.")
            return []
        
        print("🎯 Processing trajectory with sensor fusion...")
        
        # Apply sensor-based filtering
        filtered_poses = trajectory_filtering.apply_sensor_based_filtering(
            self.data['raw_poses'],
            self.data['frame_info'],
            self.data['sensor_data'],
            self.config['filtering']
        )
        
        self.results['filtered_poses'] = filtered_poses
        
        # Calculate statistics
        stats = data_processing.calculate_data_statistics(filtered_poses)
        self.results['statistics'] = stats
        
        print(f"✅ Trajectory processing complete: {len(filtered_poses)} poses")
        return filtered_poses
    
    def save_results(self, output_path):
        """
        Save processed results to file.
        
        Args:
            output_path (str): Path to save the results
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 'filtered_poses' not in self.results:
            print("❌ No processed results to save. Run process_trajectory() first.")
            return False
        
        return data_processing.save_processed_data(self.results['filtered_poses'], output_path)
    
    def calculate_synchronization_parameters(self, video_path=None):
        """
        Calculate parameters for video synchronization.
        
        Args:
            video_path (str, optional): Path to video file
            
        Returns:
            dict: Synchronization parameters
        """
        if 'filtered_poses' not in self.results:
            print("❌ No processed results available.")
            return {}
        
        poses = self.results['filtered_poses']
        
        # Extract timing information
        if poses and 'time' in poses[0]:
            start_time = poses[0]['time']
            end_time = poses[-1]['time']
            data_duration = end_time - start_time
            
            print(f"⏱️ Temporal analysis:")
            print(f"   • Data start: {start_time:.1f}s")
            print(f"   • Data end: {end_time:.1f}s")
            print(f"   • Data duration: {data_duration:.1f}s")
            
            sync_params = {
                'data_duration': data_duration,
                'start_time': start_time,
                'end_time': end_time,
                'frame_count': len(poses)
            }
            
            # Compare with video duration if available
            if video_path:
                video_duration = get_video_duration(video_path)
                if video_duration:
                    sync_params['video_duration'] = video_duration
                    sync_params['duration_match'] = abs(data_duration - video_duration) < 2
                    
                    print(f"📹 Video duration: {video_duration:.1f}s")
                    
                    if sync_params['duration_match']:
                        target_duration = data_duration
                        print("✅ Data and video durations are similar")
                    else:
                        target_duration = video_duration
                        print("⚠️ Data and video durations differ - using video duration")
                else:
                    target_duration = data_duration
                    print("⚠️ Could not get video duration - using data duration")
            else:
                target_duration = data_duration
                print("📊 No video provided - using data duration")
            
            # Calculate FPS for synchronization
            fps = len(poses) / target_duration
            sync_params['target_duration'] = target_duration
            sync_params['fps'] = fps
            
            print(f"🎯 Synchronization: {len(poses)} frames in {target_duration:.1f}s = {fps:.1f} FPS")
            
        else:
            # Fallback if no timestamps
            sync_params = {
                'target_duration': 30,
                'fps': len(poses) / 30,
                'frame_count': len(poses),
                'fallback': True
            }
            print("⚠️ No timestamps found - using fallback parameters")
        
        return sync_params
    
    def visualize_results(self, data_folder_name, video_path=None):
        """
        Visualize the processed trajectory results.
        
        Args:
            data_folder_name (str): Name of the data folder
            video_path (str, optional): Path to video file for synchronization
        """
        if 'filtered_poses' not in self.results:
            print("❌ No processed results to visualize.")
            return
        
        poses = self.results['filtered_poses']
        
        print("\n" + "="*60)
        print("🎬 STARTING ADVANCED VISUALIZATION")
        
        # Calculate synchronization parameters
        sync_params = self.calculate_synchronization_parameters(video_path)
        
        # Start synchronized video if available
        video_thread = None
        if video_path and self.config['visualization']['auto_play_video']:
            print(f"🎥 Starting synchronized video: {os.path.basename(video_path)}")
            video_duration = sync_params.get('target_duration', 30) + self.config['visualization']['video_sync_buffer']
            video_thread = play_video_synchronized(video_path, video_duration)
        
        # Start visualization
        try:
            fps = sync_params.get('fps', self.config['visualization']['fps'])
            print(f"🎬 Starting visualization at {fps:.1f} FPS...")
            visualize_poses_directly(poses, data_folder_name, fps=fps)
            
        except KeyboardInterrupt:
            print("\n⏹️ Visualization interrupted by user")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Check that visualization modules are available")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            print("💡 Try closing other matplotlib windows")
    
    def run_full_pipeline(self, input_folder):
        """
        Run the complete trajectory extraction pipeline.
        
        Args:
            input_folder (str): Path to input data folder
            
        Returns:
            bool: True if successful, False otherwise
        """
        folder_name = os.path.basename(input_folder)
        print(f"📁 Processing folder: {folder_name}")
        
        # Find required files
        jsonl_files = [f for f in os.listdir(input_folder) if f.endswith('.jsonl')]
        if not jsonl_files:
            print("❌ No JSONL file found")
            return False
        
        video_path = find_video_file(input_folder)
        if video_path:
            print(f"🎥 Video found: {os.path.basename(video_path)}")
        else:
            print("⚠️ No video file found")
        
        # Process data
        jsonl_path = os.path.join(input_folder, jsonl_files[0])
        
        if not self.load_data(jsonl_path):
            return False
        
        processed_poses = self.process_trajectory()
        if not processed_poses:
            return False
        
        # Save results
        results_folder = os.path.join('results', folder_name)
        os.makedirs(results_folder, exist_ok=True)
        output_path = os.path.join(results_folder, 'poses_2d_conservative.json')
        
        if not self.save_results(output_path):
            return False
        
        print(f"📊 Processed {len(processed_poses)} poses successfully")
        
        # Visualize results
        self.visualize_results(folder_name, video_path)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"💾 Results saved to: {output_path}")
        print(f"🎬 Visualization finished")
        print("="*60)
        
        return True


def main():
    """Main function to run the trajectory extractor."""
    print("🎯 CONSERVATIVE TRAJECTORY EXTRACTOR WITH SENSOR FUSION")
    print("="*60)
    print("🧠 Features:")
    print("   • Frame filtering by feature count (minimum 10)")
    print("   • Accelerometer + gyroscope movement detection")
    print("   • Gyroscope-directed movement (compass-based)")
    print("   • Camera rotation vs translation distinction")
    print("   • Conservative movement filtering")
    print("   • Temporal synchronization with video")
    print()
    
    # Select input folder
    input_folder = select_data_folder()
    if not input_folder:
        print("❌ No folder selected")
        return 1
    
    # Create and run trajectory extractor
    extractor = TrajectoryExtractor()
    
    try:
        success = extractor.run_full_pipeline(input_folder)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)