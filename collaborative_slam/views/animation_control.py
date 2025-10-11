"""
Animation Control Module

This module provides animation control functionality including playback management,
user input handling, and frame-by-frame animation logic.
"""

import matplotlib.animation as animation


class AnimationController:
    """
    Controls animation playbook, user input, and frame updates.
    """
    
    def __init__(self, temporal_synchronizer, animation_timer):
        """
        Initialize animation controller.
        
        Args:
            temporal_synchronizer: TemporalSynchronizer instance
            animation_timer: AnimationTimer instance
        """
        self.synchronizer = temporal_synchronizer
        self.timer = animation_timer
        self.current_frame_idx = 0
        self.is_paused = False
        self.manual_frame_control = False  # For manual frame stepping
        
    def get_current_frame_index(self):
        """
        Get the current frame index based on timing or manual control.
        
        Returns:
            int: Current frame index
        """
        if self.manual_frame_control:
            # Use manually set frame index
            return self.current_frame_idx
        elif self.is_paused:
            # Keep current frame when paused
            return self.current_frame_idx
        else:
            # Use temporal synchronization
            elapsed_time = self.timer.get_elapsed_time()
            frame_idx = self.synchronizer.get_frame_for_elapsed_time(elapsed_time)
            self.current_frame_idx = frame_idx
            return frame_idx
    
    def pause_resume(self):
        """Toggle pause/resume state."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()
    
    def pause(self):
        """Pause the animation."""
        self.is_paused = True
        self.timer.pause()
        self.manual_frame_control = True
        
    def resume(self):
        """Resume the animation."""
        self.is_paused = False
        self.timer.resume()
        self.manual_frame_control = False
        
    def restart(self):
        """Restart the animation from the beginning."""
        self.current_frame_idx = 0
        self.is_paused = False
        self.manual_frame_control = False
        self.timer.reset()
        self.timer.start()
        
    def previous_frame(self):
        """Go to previous frame (manual control)."""
        self.manual_frame_control = True
        self.is_paused = True
        self.timer.pause()
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        
    def next_frame(self):
        """Go to next frame (manual control)."""
        self.manual_frame_control = True
        self.is_paused = True
        self.timer.pause()
        total_frames = len(self.synchronizer.poses_data)
        self.current_frame_idx = min(total_frames - 1, self.current_frame_idx + 1)
        
    def handle_key_press(self, event):
        """
        Handle keyboard input for animation control.
        
        Args:
            event: Matplotlib key press event
        """
        if event.key == ' ':  # Space - pause/resume
            self.pause_resume()
        elif event.key == 'left':  # Left arrow - previous frame
            self.previous_frame()
        elif event.key == 'right':  # Right arrow - next frame
            self.next_frame()
        elif event.key == 'r':  # R - restart
            self.restart()
        elif event.key == 'escape':  # ESC - close
            import matplotlib.pyplot as plt
            plt.close('all')


class TrajectoryAnimator:
    """
    Main animator class that coordinates all animation components.
    """
    
    def __init__(self, visualization_components, animation_controller, 
                 check_interval_ms=20):
        """
        Initialize trajectory animator.
        
        Args:
            visualization_components (dict): Dictionary of visualization components
            animation_controller: AnimationController instance
            check_interval_ms (int): Animation check interval in milliseconds
        """
        self.components = visualization_components
        self.controller = animation_controller
        self.check_interval = check_interval_ms
        self.animation = None
        
        # Extract components
        self.trajectory_plotter = self.components['trajectory_plotter']
        self.orientation_plotter = self.components['orientation_plotter']
        self.layout = self.components['layout']
        
        # Extract pose data for convenience
        self.poses_data = self.controller.synchronizer.poses_data
        self.orientations = [p.get("orientacion", 0) for p in self.poses_data]
        self.initial_orientations = [p.get("orientacion_inicial", 0) for p in self.poses_data]
        self.timestamps = self.controller.synchronizer.timestamps
        
    def _animate_frame(self, frame_number):
        """
        Animation callback function.
        
        Args:
            frame_number: Frame number from matplotlib animation (unused, we use timing)
            
        Returns:
            tuple: Updated plot elements
        """
        # Get current frame index from controller
        frame_idx = self.controller.get_current_frame_index()
        
        # Handle end of animation
        if frame_idx >= len(self.poses_data):
            if not self.controller.manual_frame_control:
                self.controller.restart()
            frame_idx = 0
        
        # Get current data
        current_orientation = self.orientations[frame_idx] if frame_idx < len(self.orientations) else 0
        initial_orientation = self.initial_orientations[frame_idx] if frame_idx < len(self.initial_orientations) else 0
        current_timestamp = self.timestamps[frame_idx] if frame_idx < len(self.timestamps) else None
        
        # Update visualization components
        self.trajectory_plotter.update_frame(frame_idx, current_orientation, 
                                           initial_orientation, current_timestamp)
        self.orientation_plotter.update_frame(frame_idx)
        
        # Return updated elements (for blitting if enabled)
        return (self.trajectory_plotter.line_trail, 
                self.trajectory_plotter.line_current,
                self.trajectory_plotter.arrow_current,
                self.trajectory_plotter.arrow_initial)
    
    def start_animation(self):
        """Start the animation."""
        # Start the timer
        self.controller.timer.start()
        
        # Connect keyboard event handler
        self.layout.fig.canvas.mpl_connect('key_press_event', 
                                          self.controller.handle_key_press)
        
        # Create matplotlib animation
        self.animation = animation.FuncAnimation(
            self.layout.fig, 
            self._animate_frame,
            interval=self.check_interval,
            blit=False,
            repeat=True,
            cache_frame_data=False
        )
        
        # Finalize layout and show
        self.layout.finalize_layout()
        
        print(f"⏱️ Animation started:")
        print(f"   • Check interval: {self.check_interval}ms")
        print(f"   • Total frames: {len(self.poses_data)}")
        print(f"   • Duration: {self.controller.synchronizer.timing_params['total_duration']:.1f}s")
        print(f"   • Controls: Space (pause), ← → (frame), R (restart), ESC (close)")
        
        try:
            self.layout.show()
        except KeyboardInterrupt:
            print("\n⏹️ Animation interrupted by user")
            self.layout.close()
        except Exception as e:
            print(f"\n❌ Animation error: {e}")
            self.layout.close()
            raise
            
        return self.animation