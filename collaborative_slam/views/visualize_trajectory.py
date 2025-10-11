"""
Advanced Trajectory Visualizer

A modular and well-documented trajectory visualization system with temporal synchronization.

This module provides advanced visualization capabilities for trajectory data including:
- Real-time temporal synchronization using timestamps
- Interactive animation control (pause, step, restart)
- Orientation visualization with directional arrows
- Statistics and timing information display
- Modular component architecture for maintainability

Features:
- Temporal synchronization based on real timestamps
- Dynamic interval calculation for smooth playback
- Interactive controls (keyboard shortcuts)
- Multi-panel layout with trajectory, orientation, and statistics
- Configurable animation parameters
- Error handling and graceful cleanup

Author: TFM Project  
Version: 2.0 - Modular and Clean
"""

import os
import json
import numpy as np

# Import visualization modules
try:
    from .temporal_sync import TemporalSynchronizer, AnimationTimer
    from .visualization_components import (
        TrajectoryPlotter, OrientationPlotter, StatisticsPanel, VisualizationLayout
    )
    from .animation_control import AnimationController, TrajectoryAnimator
except ImportError:
    # Fallback for direct execution
    from utils.temporal_sync import TemporalSynchronizer, AnimationTimer
    from utils.visualization_components import (
        TrajectoryPlotter, OrientationPlotter, StatisticsPanel, VisualizationLayout
    )
    from utils.animation_control import AnimationController, TrajectoryAnimator


class AdvancedTrajectoryVisualizer:
    """
    Main class for advanced trajectory visualization with temporal synchronization.
    """
    
    def __init__(self, config=None):
        """
        Initialize the trajectory visualizer.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or self._default_config()
        self.components = {}
        self.data_loaded = False
        
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'figure_size': (15, 10),
            'animation_interval_ms': 20,  # Check interval for temporal sync
            'enable_temporal_sync': True,
            'show_statistics': True,
            'show_orientation_plot': True,
            'arrow_scale_factor': 0.1,
            'trail_alpha': 0.6,
            'current_marker_size': 8
        }
    
    def load_poses_from_data(self, poses_data, data_folder_name="Unknown"):
        """
        Load pose data directly from memory.
        
        Args:
            poses_data (list): List of pose dictionaries
            data_folder_name (str): Name for display purposes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not poses_data:
                print("❌ No pose data provided")
                return False
            
            self.poses_data = poses_data
            self.data_folder_name = data_folder_name
            self.data_loaded = True
            
            print(f"✅ Loaded {len(poses_data)} poses for visualization")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pose data: {e}")
            return False
    
    def load_poses_from_json(self, json_path):
        """
        Load pose data from JSON file.
        
        Args:
            json_path (str): Path to JSON file with pose data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(json_path, 'r') as f:
                poses_data = json.load(f)
            
            data_folder_name = os.path.basename(os.path.dirname(json_path))
            return self.load_poses_from_data(poses_data, data_folder_name)
            
        except Exception as e:
            print(f"❌ Error loading JSON file {json_path}: {e}")
            return False
    
    def _validate_pose_data(self):
        """
        Validate loaded pose data.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not self.data_loaded or not self.poses_data:
            print("❌ No pose data loaded")
            return False
        
        # Check for required fields
        required_fields = ['x', 'y']
        for i, pose in enumerate(self.poses_data[:5]):  # Check first 5 poses
            for field in required_fields:
                if field not in pose:
                    print(f"❌ Missing required field '{field}' in pose {i}")
                    return False
        
        print(f"✅ Pose data validation passed ({len(self.poses_data)} poses)")
        return True
    
    def _create_visualization_components(self, fps=25):
        """
        Create all visualization components.
        
        Args:
            fps (float): Target frames per second for display info
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract data arrays
            positions = np.array([[p["x"], p["y"]] for p in self.poses_data])
            orientations = [p.get("orientacion", 0) for p in self.poses_data]
            initial_orientations = [p.get("orientacion_inicial", 0) for p in self.poses_data]
            
            # Create temporal synchronizer
            temporal_sync = TemporalSynchronizer(self.poses_data)
            timing_info = temporal_sync.get_timing_info()
            
            print(f"⏱️ Temporal synchronization:")
            print(f"   • Total duration: {timing_info['total_duration']:.1f}s")
            print(f"   • Average interval: {timing_info['average_interval']:.1f}ms")
            print(f"   • Frame count: {timing_info['frame_count']}")
            
            # Create layout
            layout = VisualizationLayout(self.config['figure_size'])
            ax_main, ax_orientation, ax_stats = layout.get_axes()
            
            # Create visualization components
            trajectory_plotter = TrajectoryPlotter(ax_main, positions, self.data_folder_name)
            orientation_plotter = OrientationPlotter(ax_orientation, orientations, initial_orientations)
            stats_panel = StatisticsPanel(ax_stats, self.poses_data, timing_info)
            
            # Create animation control
            animation_timer = AnimationTimer()
            animation_controller = AnimationController(temporal_sync, animation_timer)
            
            # Store components
            self.components = {
                'layout': layout,
                'trajectory_plotter': trajectory_plotter,
                'orientation_plotter': orientation_plotter,
                'stats_panel': stats_panel,
                'temporal_sync': temporal_sync,
                'animation_timer': animation_timer,
                'animation_controller': animation_controller
            }
            
            print("✅ Visualization components created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization components: {e}")
            return False
    
    def visualize(self, fps=25):
        """
        Start the advanced visualization.
        
        Args:
            fps (float): Target frames per second (for display info only)
            
        Returns:
            object: Animation object if successful, None otherwise
        """
        if not self._validate_pose_data():
            return None
        
        if not self._create_visualization_components(fps):
            return None
        
        print(f"🎬 Starting advanced visualization...")
        print(f"   • Data: {self.data_folder_name}")
        print(f"   • Temporal sync: {'Enabled' if self.config['enable_temporal_sync'] else 'Disabled'}")
        
        try:
            # Create animator
            animator = TrajectoryAnimator(
                self.components, 
                self.components['animation_controller'],
                self.config['animation_interval_ms']
            )
            
            # Start animation
            animation_obj = animator.start_animation()
            
            print("✅ Visualization completed")
            return animation_obj
            
        except KeyboardInterrupt:
            print("\n⏹️ Visualization interrupted by user")
            self._cleanup()
            return None
        except Exception as e:
            print(f"\n❌ Visualization error: {e}")
            self._cleanup()
            raise
    
    def _cleanup(self):
        """Clean up visualization resources."""
        if 'layout' in self.components:
            self.components['layout'].close()
        print("🧹 Visualization cleanup completed")


# Convenience functions for backward compatibility
def create_advanced_visualization(filtered_poses, data_folder, fps=25):
    """
    Create advanced visualization (backward compatibility function).
    
    Args:
        filtered_poses (list): List of pose data
        data_folder (str): Data folder name
        fps (float): Target FPS for display
        
    Returns:
        object: Animation object if successful, None otherwise
    """
    visualizer = AdvancedTrajectoryVisualizer()
    
    if not visualizer.load_poses_from_data(filtered_poses, data_folder):
        return None
    
    return visualizer.visualize(fps)


def visualize_poses_directly(poses_data, data_folder_name="Unknown", fps=25):
    """
    Visualize poses directly from data (main interface function).
    
    Args:
        poses_data (list): List of pose data
        data_folder_name (str): Name for display purposes
        fps (float): Target FPS for display
        
    Returns:
        object: Animation object if successful, None otherwise
    """
    print(f"🎬 Creating advanced visualization for {len(poses_data)} poses at {fps:.1f} FPS...")
    
    visualizer = AdvancedTrajectoryVisualizer()
    
    if not visualizer.load_poses_from_data(poses_data, data_folder_name):
        return None
    
    return visualizer.visualize(fps)


def visualize_from_json_file(json_path):
    """
    Load and visualize poses from JSON file.
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        object: Animation object if successful, None otherwise
    """
    print(f"📂 Loading and visualizing poses from {json_path}...")
    
    visualizer = AdvancedTrajectoryVisualizer()
    
    if not visualizer.load_poses_from_json(json_path):
        return None
    
    return visualizer.visualize()


def load_poses_from_json(json_path):
    """
    Load poses from JSON file (utility function).
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        list: Pose data if successful, None otherwise
    """
    try:
        with open(json_path, 'r') as f:
            poses = json.load(f)
        print(f"✅ Loaded {len(poses)} poses from {os.path.basename(json_path)}")
        return poses
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return None


# Main execution for testing
if __name__ == "__main__":
    print("🎯 ADVANCED TRAJECTORY VISUALIZER")
    print("="*50)
    print("This is a test run of the modular visualization system.")
    print("For normal usage, import and use the functions in your scripts.")
    print()
    
    # Example usage
    print("Example usage:")
    print("  from views.visualize_trajectory import visualize_poses_directly")
    print("  visualize_poses_directly(poses_data, 'FolderName', fps=30)")
    print()
    print("Or:")
    print("  from views.visualize_trajectory import visualize_from_json_file") 
    print("  visualize_from_json_file('path/to/poses.json')")