"""
Visualization Components Module

This module provides specialized components for trajectory visualization,
including trajectory plots, orientation displays, and statistics panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


class TrajectoryPlotter:
    """
    Handles the main trajectory visualization plot.
    """
    
    def __init__(self, ax, positions, data_folder_name):
        """
        Initialize trajectory plotter.
        
        Args:
            ax: Matplotlib axis for plotting
            positions (np.array): Array of [x, y] positions
            data_folder_name (str): Name of data folder for title
        """
        self.ax = ax
        self.positions = positions
        self.data_folder_name = data_folder_name
        self._setup_plot()
        self._create_plot_elements()
        
    def _setup_plot(self):
        """Setup the main plot area."""
        # Calculate plot limits with margin
        x_min, x_max = self.positions[:, 0].min(), self.positions[:, 0].max()
        y_min, y_max = self.positions[:, 1].min(), self.positions[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = max(x_range, y_range) * 0.1
        
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        
        # Store dimensions for arrow scaling
        self.arrow_length = min(x_range, y_range) * 0.1
        
    def _create_plot_elements(self):
        """Create the visual elements for the plot."""
        # Trajectory line
        self.line_trail, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2, 
                                       label='Trajectory')
        
        # Current position marker
        self.line_current, = self.ax.plot([], [], 'ro', markersize=8, 
                                         label='Current position')
        
        # Orientation arrows
        self.arrow_current = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                            arrowprops=dict(arrowstyle='->', 
                                                          color='red', 
                                                          lw=3, alpha=0.8))
        
        self.arrow_initial = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                            arrowprops=dict(arrowstyle='->', 
                                                          color='green', 
                                                          lw=2, alpha=0.6))
        
        # Legend
        self.ax.legend(loc='upper right')
        
    def update_frame(self, frame_idx, current_orientation, initial_orientation, 
                    current_timestamp=None):
        """
        Update the plot for the current frame.
        
        Args:
            frame_idx (int): Current frame index
            current_orientation (float): Current orientation in radians
            initial_orientation (float): Initial orientation in radians
            current_timestamp (float, optional): Current timestamp
        """
        if frame_idx >= len(self.positions):
            return
            
        # Update trajectory trail
        trail_x = self.positions[:frame_idx+1, 0]
        trail_y = self.positions[:frame_idx+1, 1]
        self.line_trail.set_data(trail_x, trail_y)
        
        # Update current position
        current_pos = self.positions[frame_idx]
        self.line_current.set_data([current_pos[0]], [current_pos[1]])
        
        # Update current orientation arrow (red)
        end_x = current_pos[0] + self.arrow_length * np.cos(current_orientation)
        end_y = current_pos[1] + self.arrow_length * np.sin(current_orientation)
        self.arrow_current.set_position((current_pos[0], current_pos[1]))
        self.arrow_current.xy = (end_x, end_y)
        
        # Update initial orientation arrow (green - reference)
        end_x_initial = current_pos[0] + self.arrow_length * 0.7 * np.cos(initial_orientation)
        end_y_initial = current_pos[1] + self.arrow_length * 0.7 * np.sin(initial_orientation)
        self.arrow_initial.set_position((current_pos[0], current_pos[1]))
        self.arrow_initial.xy = (end_x_initial, end_y_initial)
        
        # Update title
        orientation_degrees = np.degrees(current_orientation)
        timestamp_str = f" - Time: {current_timestamp:.1f}s" if current_timestamp else ""
        
        self.ax.set_title(
            f'Frame {frame_idx+1}/{len(self.positions)}{timestamp_str} - '
            f'Orientation: {orientation_degrees:.1f}° - {self.data_folder_name}',
            fontsize=14, fontweight='bold'
        )


class OrientationPlotter:
    """
    Handles the orientation evolution plot.
    """
    
    def __init__(self, ax, orientations, initial_orientations):
        """
        Initialize orientation plotter.
        
        Args:
            ax: Matplotlib axis for plotting
            orientations (list): List of current orientations
            initial_orientations (list): List of initial orientations
        """
        self.ax = ax
        self.orientations = orientations
        self.initial_orientations = initial_orientations
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup the orientation plot."""
        frames_range = range(len(self.orientations))
        orientations_degrees = [np.degrees(o) for o in self.orientations]
        initial_orientations_degrees = [np.degrees(o) for o in self.initial_orientations]
        
        # Plot orientation evolution
        self.ax.plot(frames_range, orientations_degrees, 'r-', alpha=0.8, 
                    label='Current orientation')
        self.ax.plot(frames_range, initial_orientations_degrees, 'g--', alpha=0.6, 
                    label='Initial orientation')
        
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Orientation (degrees)')
        self.ax.set_title('Orientation Evolution')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Store for frame indicator
        self.frame_indicator = None
        
    def update_frame(self, frame_idx):
        """
        Update the orientation plot for the current frame.
        
        Args:
            frame_idx (int): Current frame index
        """
        # Remove previous frame indicator
        if self.frame_indicator is not None:
            self.frame_indicator.remove()
            
        # Add new frame indicator
        self.frame_indicator = self.ax.axvline(x=frame_idx, color='blue', 
                                             alpha=0.7, linestyle='--')


class StatisticsPanel:
    """
    Handles the statistics display panel.
    """
    
    def __init__(self, ax, filtered_poses, timing_info):
        """
        Initialize statistics panel.
        
        Args:
            ax: Matplotlib axis for the panel
            filtered_poses (list): List of filtered poses
            timing_info (dict): Timing information from synchronizer
        """
        self.ax = ax
        self.filtered_poses = filtered_poses
        self.timing_info = timing_info
        self._setup_panel()
        
    def _setup_panel(self):
        """Setup the statistics panel."""
        # Calculate statistics
        positions = np.array([[p["x"], p["y"]] for p in self.filtered_poses])
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[i] - positions[i-1])
        
        # Calculate ranges
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        # Calculate orientation statistics
        orientations = [p.get("orientacion", 0) for p in self.filtered_poses]
        orientations_degrees = [np.degrees(o) for o in orientations]
        orientation_range = max(orientations_degrees) - min(orientations_degrees)
        
        # Create statistics text
        stats_text = f"""Statistics:

• Total frames: {len(self.filtered_poses)}
• Total distance: {total_distance:.1f} px
• X range: {x_range:.1f} px  
• Y range: {y_range:.1f} px
• Orientation range: {orientation_range:.1f}°

Temporal Info:
• Duration: {self.timing_info['total_duration']:.1f}s
• Avg interval: {self.timing_info['average_interval']:.1f}ms
• Start time: {self.timing_info['start_time']:.1f}s
• End time: {self.timing_info['end_time']:.1f}s

Controls:
• Space: Pause/Continue
• ← →: Previous/Next frame  
• R: Restart
• ESC: Close"""
        
        # Display text
        self.ax.text(0.05, 0.95, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')


class VisualizationLayout:
    """
    Manages the overall layout of the visualization.
    """
    
    def __init__(self, figsize=(15, 10)):
        """
        Initialize visualization layout.
        
        Args:
            figsize (tuple): Figure size (width, height)
        """
        self.fig = plt.figure(figsize=figsize)
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
        self._create_subplots()
        
    def _create_subplots(self):
        """Create the subplot structure."""
        self.ax_main = self.fig.add_subplot(self.gs[0, :])        # Main trajectory plot
        self.ax_orientation = self.fig.add_subplot(self.gs[1, 0])  # Orientation plot
        self.ax_stats = self.fig.add_subplot(self.gs[1, 1])       # Statistics panel
        
    def get_axes(self):
        """
        Get all axes for the visualization.
        
        Returns:
            tuple: (main_axis, orientation_axis, stats_axis)
        """
        return self.ax_main, self.ax_orientation, self.ax_stats
        
    def finalize_layout(self):
        """Finalize the layout and prepare for display."""
        plt.tight_layout()
        
    def show(self):
        """Display the visualization."""
        plt.show()
        
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)