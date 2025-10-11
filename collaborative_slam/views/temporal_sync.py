"""
Temporal Synchronization Module

This module handles temporal synchronization between trajectory data and external media,
providing utilities for calculating timing parameters and managing time-based playback.
"""

import time
import numpy as np


class TemporalSynchronizer:
    """
    Handles temporal synchronization between trajectory animation and external media.
    """
    
    def __init__(self, poses_data):
        """
        Initialize synchronizer with pose data.
        
        Args:
            poses_data (list): List of poses with temporal information
        """
        self.poses_data = poses_data
        self.timestamps = self._extract_timestamps()
        self.timing_params = self._calculate_timing_parameters()
        
    def _extract_timestamps(self):
        """Extract timestamps from pose data."""
        timestamps = []
        for i, pose in enumerate(self.poses_data):
            timestamp = pose.get("time", i * 0.033)  # Fallback to ~30fps
            timestamps.append(timestamp)
        return timestamps
    
    def _calculate_timing_parameters(self):
        """Calculate timing parameters for synchronization."""
        if len(self.timestamps) < 2:
            return {
                'total_duration': 30.0,
                'average_interval': 33.33,  # ms
                'dynamic_intervals': [33.33] * len(self.poses_data)
            }
        
        # Calculate dynamic intervals based on real timestamps
        intervals = []
        for i in range(1, len(self.timestamps)):
            interval = (self.timestamps[i] - self.timestamps[i-1]) * 1000  # Convert to ms
            # Clamp intervals to reasonable bounds
            clamped_interval = max(10, min(interval, 200))
            intervals.append(clamped_interval)
        
        # Add initial interval
        intervals.insert(0, intervals[0] if intervals else 33.33)
        
        total_duration = self.timestamps[-1] - self.timestamps[0]
        average_interval = np.mean(intervals)
        
        return {
            'total_duration': total_duration,
            'average_interval': average_interval,
            'dynamic_intervals': intervals
        }
    
    def get_frame_for_elapsed_time(self, elapsed_seconds):
        """
        Get the appropriate frame index for given elapsed time.
        
        Args:
            elapsed_seconds (float): Time elapsed since animation start
            
        Returns:
            int: Frame index that should be displayed
        """
        if not self.timestamps:
            return 0
        
        # Add offset to match the first timestamp
        time_offset = self.timestamps[0]
        target_time = elapsed_seconds + time_offset
        
        # Find closest frame
        target_frame = 0
        for i, timestamp in enumerate(self.timestamps):
            if timestamp <= target_time:
                target_frame = i
            else:
                break
        
        return min(target_frame, len(self.poses_data) - 1)
    
    def get_timing_info(self):
        """Get timing information for display."""
        return {
            'total_duration': self.timing_params['total_duration'],
            'average_interval': self.timing_params['average_interval'],
            'frame_count': len(self.poses_data),
            'start_time': self.timestamps[0] if self.timestamps else 0,
            'end_time': self.timestamps[-1] if self.timestamps else 0
        }


class AnimationTimer:
    """
    Manages animation timing and playback control.
    """
    
    def __init__(self):
        """Initialize animation timer."""
        self.start_time = None
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_duration = 0
        
    def start(self):
        """Start the animation timer."""
        self.start_time = time.time()
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_duration = 0
        
    def pause(self):
        """Pause the animation timer."""
        if not self.is_paused and self.start_time is not None:
            self.is_paused = True
            self.pause_start_time = time.time()
            
    def resume(self):
        """Resume the animation timer."""
        if self.is_paused and self.pause_start_time is not None:
            self.total_pause_duration += time.time() - self.pause_start_time
            self.is_paused = False
            self.pause_start_time = None
            
    def get_elapsed_time(self):
        """
        Get elapsed time since animation start (excluding pause time).
        
        Returns:
            float: Elapsed time in seconds, or 0 if not started
        """
        if self.start_time is None:
            return 0
            
        current_time = time.time()
        elapsed = current_time - self.start_time - self.total_pause_duration
        
        # If currently paused, don't count the current pause duration
        if self.is_paused and self.pause_start_time is not None:
            elapsed -= (current_time - self.pause_start_time)
            
        return max(0, elapsed)
        
    def reset(self):
        """Reset the animation timer."""
        self.start_time = None
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_duration = 0