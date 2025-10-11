"""
Trajectory Filtering Module

This module provides advanced filtering algorithms for trajectory data,
including sensor-based filtering and movement classification.
"""

import numpy as np
from scipy.ndimage import median_filter
try:
    from .sensor_processing import synchronize_sensors_with_frames
except ImportError:
    from sensor_processing import synchronize_sensors_with_frames


def filter_low_feature_frames(entries, min_features=10):
    """
    Filter out frames with insufficient feature points.
    
    Args:
        entries (list): Raw JSONL entries
        min_features (int): Minimum number of features required
        
    Returns:
        tuple: (filtered_poses, frame_info) with sufficient features
    """
    filtered_poses = []
    frame_info = []
    discarded_count = 0
    
    print(f"🔍 Filtering frames with minimum {min_features} features...")
    
    for entry in entries:
        if "frames" in entry:
            for frame in entry["frames"]:
                if frame.get("features"):
                    points = [feat["point"] for feat in frame["features"] if "point" in feat]
                    
                    if len(points) >= min_features:
                        points_np = np.array(points)
                        centroid = points_np.mean(axis=0)[:2]  # X, Y only
                        
                        filtered_poses.append({
                            "x": float(centroid[0]), 
                            "y": float(centroid[1])
                        })
                        
                        frame_info.append({
                            "num_points": len(points),
                            "time": entry.get('time', 0)
                        })
                    else:
                        discarded_count += 1
                        if discarded_count <= 5:  # Show first few discarded frames
                            print(f"   Frame with {len(points)} features DISCARDED")
    
    if discarded_count > 5:
        print(f"   ... and {discarded_count - 5} more frames discarded")
    
    print(f"✅ Kept {len(filtered_poses)} frames, discarded {discarded_count} frames")
    return filtered_poses, frame_info


def apply_initial_smoothing(positions, window_size=None):
    """
    Apply light smoothing to initial trajectory data.
    
    Args:
        positions (np.array): Array of [x, y] positions
        window_size (int, optional): Smoothing window size
        
    Returns:
        np.array: Smoothed positions
    """
    if window_size is None:
        window_size = min(5, len(positions) // 8)
    
    if window_size >= 3 and len(positions) > window_size:
        print(f"🔧 Applying initial smoothing (window size: {window_size})")
        x_smooth = median_filter(positions[:, 0], size=window_size)
        y_smooth = median_filter(positions[:, 1], size=window_size)
        return np.column_stack([x_smooth, y_smooth])
    else:
        print("⏭️ Skipping smoothing (insufficient data)")
        return positions


def find_stable_initial_position(positions, max_frames=5):
    """
    Find the most stable initial position to avoid jumps.
    
    Args:
        positions (np.array): Array of initial positions
        max_frames (int): Maximum number of initial frames to consider
        
    Returns:
        np.array: Stable initial position [x, y]
    """
    initial_frames = min(max_frames, len(positions))
    initial_positions = positions[:initial_frames]
    
    if len(initial_positions) > 1:
        variations = []
        for i in range(1, len(initial_positions)):
            variation = np.linalg.norm(initial_positions[i] - initial_positions[i-1])
            variations.append(variation)
        
        if variations:
            # Use frame with minimum variation as starting point
            stable_index = np.argmin(variations) + 1
            stable_position = initial_positions[stable_index]
            print(f"📍 Stable initial position found at frame {stable_index}: ({stable_position[0]:.1f}, {stable_position[1]:.1f})")
            return stable_position
    
    stable_position = positions[0]
    print(f"📍 Using first position as initial: ({stable_position[0]:.1f}, {stable_position[1]:.1f})")
    return stable_position


def apply_sensor_based_filtering(raw_poses, frame_info, sensor_data, config=None):
    """
    Apply intelligent filtering using sensor data to distinguish camera rotation from real movement.
    
    Args:
        raw_poses (list): Raw pose data
        frame_info (list): Frame information
        sensor_data (dict): Sensor readings
        config (dict, optional): Filtering configuration
        
    Returns:
        list: Filtered poses with sensor-based movement classification
    """
    if config is None:
        config = {
            'movement_threshold_high': 25,      # High threshold for confirmed translation
            'movement_threshold_uncertain': 40, # Very high threshold for uncertain movement
            'movement_scale_confirmed': 0.2,    # Scale factor for confirmed movement
            'movement_scale_uncertain': 0.05,   # Scale factor for uncertain movement
            'max_movement_per_frame': 15,       # Maximum movement per frame
            'early_frame_limit': 10,            # Frames to apply extra filtering
            'early_frame_threshold': 100        # Movement threshold for early frames
        }
    
    print("🎯 Applying sensor-based intelligent filtering...")
    
    # Synchronize sensor data with frames
    frame_sensor_info = synchronize_sensors_with_frames(frame_info, sensor_data)
    
    # Print movement analysis statistics
    movement_types = [info['movement_type'] for info in frame_sensor_info]
    stats = {
        'still': movement_types.count('still'),
        'rotation': movement_types.count('rotation'),
        'translation': movement_types.count('translation'),
        'maybe_translation': movement_types.count('maybe_translation'),
        'unknown': movement_types.count('unknown')
    }
    
    total_frames = len(movement_types)
    print(f"   📊 Movement analysis (conservative thresholds):")
    for movement_type, count in stats.items():
        percentage = (count / total_frames * 100) if total_frames > 0 else 0
        print(f"     • {movement_type.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
    
    # Convert to numpy array for processing
    positions = np.array([[p["x"], p["y"]] for p in raw_poses])
    
    # Apply initial smoothing
    positions_smooth = apply_initial_smoothing(positions)
    
    # Find stable initial position
    fixed_position = find_stable_initial_position(positions_smooth)
    
    # Apply sensor-based filtering
    filtered_poses = []
    frames_moved = 0
    frames_still = 0
    frames_rotation_blocked = 0
    
    print(f"🔄 Processing {len(positions_smooth)} frames...")
    
    for i, (pos, sensor_info) in enumerate(zip(positions_smooth, frame_sensor_info)):
        movement_type = sensor_info['movement_type']
        camera_orientation = sensor_info.get('camera_orientation', 0)
        initial_orientation = sensor_info.get('initial_orientation', 0)
        
        # Calculate orientation delta for reference
        delta_orientation = camera_orientation - initial_orientation
        
        # Create base pose data
        pose_data = {
            "x": float(fixed_position[0]),
            "y": float(fixed_position[1]),
            "orientacion": float(camera_orientation),
            "orientacion_inicial": float(initial_orientation),
            "delta_orientacion": float(delta_orientation),
            "time": float(sensor_info['frame_info']['time'])
        }
        
        if movement_type == "translation":
            # CONFIRMED REAL MOVEMENT
            distance_moved = np.linalg.norm(pos - fixed_position)
            
            # Extra filtering for early frames to prevent initial jumps
            if i < config['early_frame_limit'] and distance_moved > config['early_frame_threshold']:
                if i < 5:  # Debug for early frames
                    print(f"     Frame {i}: INITIAL JUMP BLOCKED ({distance_moved:.1f}px)")
                frames_still += 1
            elif distance_moved > config['movement_threshold_high']:
                # Apply gyroscope-directed movement
                movement_magnitude = min(distance_moved * config['movement_scale_confirmed'], 
                                       config['max_movement_per_frame'])
                
                # Calculate movement direction based on camera orientation
                dx = movement_magnitude * np.cos(camera_orientation)
                dy = movement_magnitude * np.sin(camera_orientation)
                
                # Update fixed position
                new_position = fixed_position + np.array([dx, dy])
                fixed_position = new_position
                
                pose_data["x"] = float(new_position[0])
                pose_data["y"] = float(new_position[1])
                frames_moved += 1
                
                if i < 5:  # Debug for early frames
                    orientation_degrees = np.degrees(camera_orientation)
                    print(f"     Frame {i}: TRANSLATION ({distance_moved:.1f}px) → {movement_magnitude:.1f}px toward {orientation_degrees:.0f}°")
            else:
                frames_still += 1
                
        elif movement_type == "maybe_translation":
            # UNCERTAIN MOVEMENT - Be EXTREMELY conservative, mostly treat as still
            distance_moved = np.linalg.norm(pos - fixed_position)
            
            # Balanced threshold for uncertain movement
            if (distance_moved > config['movement_threshold_uncertain'] * 1.2 and  # Slightly higher threshold
                i > 5):  # Wait a few frames to avoid early noise
                # Small movement for uncertain cases
                movement_magnitude = min(distance_moved * config['movement_scale_uncertain'] * 0.7, 
                                       config['max_movement_per_frame'] // 4)  # Moderate movement reduction
                
                dx = movement_magnitude * np.cos(camera_orientation)
                dy = movement_magnitude * np.sin(camera_orientation)
                
                new_position = fixed_position + np.array([dx, dy])
                fixed_position = new_position
                
                pose_data["x"] = float(new_position[0])
                pose_data["y"] = float(new_position[1])
                frames_moved += 1
                
                if i < 5:  # Debug for early frames
                    orientation_degrees = np.degrees(camera_orientation)
                    print(f"     Frame {i}: UNCERTAIN MOVEMENT ({distance_moved:.1f}px) → {movement_magnitude:.1f}px toward {orientation_degrees:.0f}°")
            else:
                frames_still += 1
                
        elif movement_type == "rotation":
            # CAMERA ROTATION - Don't update position
            frames_rotation_blocked += 1
            if i < 5:  # Debug for early frames
                print(f"     Frame {i}: ROTATION → position fixed")
        else:
            # STILL or UNKNOWN - Keep position fixed
            frames_still += 1
            if i < 5 and movement_type == "still":  # Debug for early frames
                print(f"     Frame {i}: STILL → position fixed")
        
        filtered_poses.append(pose_data)
    
    # Print filtering results
    print(f"   📈 Filtering results:")
    print(f"     • Frames with real movement: {frames_moved}")
    print(f"     • Frames kept still: {frames_still}")
    print(f"     • Rotations blocked: {frames_rotation_blocked}")
    print(f"     • Real movement ratio: {frames_moved/len(raw_poses)*100:.1f}%")
    
    return filtered_poses