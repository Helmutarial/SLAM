"""
Sensor Data Processing Module

This module handles the analysis and processing of sensor data (accelerometer and gyroscope)
for trajectory extraction and movement classification.
"""

import numpy as np


def analyze_sensor_data(sensor_data):
    """
    Analyze sensor data to extract meaningful statistics.
    
    Args:
        sensor_data (dict): Dictionary containing 'accelerometer' and 'gyroscope' data
        
    Returns:
        dict: Analysis results with statistics for each sensor type
    """
    analysis = {}
    
    for sensor_type, data in sensor_data.items():
        if data:
            magnitudes = [d['magnitude'] for d in data]
            analysis[sensor_type] = {
                'count': len(data),
                'mean_magnitude': np.mean(magnitudes),
                'std_magnitude': np.std(magnitudes),
                'max_magnitude': np.max(magnitudes),
                'min_magnitude': np.min(magnitudes)
            }
        else:
            analysis[sensor_type] = {
                'count': 0,
                'mean_magnitude': 0,
                'std_magnitude': 0,
                'max_magnitude': 0,
                'min_magnitude': 0
            }
    
    return analysis


def detect_movement_type(accel_data, gyro_data, thresholds=None):
    """
    Detect movement type based on accelerometer and gyroscope data.
    
    Args:
        accel_data (list): Accelerometer readings within time window
        gyro_data (list): Gyroscope readings within time window  
        thresholds (dict, optional): Custom thresholds for classification
        
    Returns:
        str: Movement type ('still', 'rotation', 'translation', 'maybe_translation', 'unknown')
    """
    if not accel_data or not gyro_data:
        return "unknown"
    
    # EVEN LESS SENSITIVE thresholds - otro poquitín menos
    if thresholds is None:
        thresholds = {
            'accel_threshold': 1.8,      # Un poquitín más alto todavía
            'gyro_threshold': 0.6,       # Activity threshold for rotation
            'variance_threshold': 0.8,   # Un poquito más de varianza necesaria
            'min_samples': 3             # Back to 3 samples for easier detection
        }
    
    # Analyze accelerometer data (detects physical translation)
    # Filter out gravity (~9.8 m/s²)
    accel_magnitudes = [abs(d['magnitude'] - 9.8) for d in accel_data]
    accel_activity = np.mean(accel_magnitudes) if accel_magnitudes else 0
    accel_variance = np.var(accel_magnitudes) if len(accel_magnitudes) > 1 else 0
    
    # Analyze gyroscope data (detects camera rotation)
    gyro_magnitudes = [d['magnitude'] for d in gyro_data]
    gyro_activity = np.mean(gyro_magnitudes) if gyro_magnitudes else 0
    
    # VERY CONSERVATIVE classification logic - prefer "still" over movement
    if (gyro_activity > thresholds['gyro_threshold'] and 
        accel_activity < thresholds['accel_threshold'] * 0.3):  # More strict rotation detection
        return "rotation"  # Pure camera rotation
    elif (accel_activity > thresholds['accel_threshold'] and 
          accel_variance > thresholds['variance_threshold'] and
          len(accel_magnitudes) > thresholds['min_samples']):
        return "translation"  # Confirmed physical movement (very strict)
    elif accel_activity > thresholds['accel_threshold'] * 0.6:  # Un poquitín más alto todavía
        return "maybe_translation"  # Uncertain movement with even less permissive threshold
    else:
        return "still"  # Default to still for most cases


def calculate_camera_orientation(gyro_data, initial_orientation=None, noise_threshold=0.1):
    """
    Calculate camera orientation using gyroscope data integration.
    
    Args:
        gyro_data (list): Gyroscope readings
        initial_orientation (float, optional): Previous orientation as reference
        noise_threshold (float): Threshold for filtering gyroscope noise
        
    Returns:
        float: Camera orientation in radians, normalized to [-π, π]
    """
    if not gyro_data:
        return initial_orientation if initial_orientation is not None else 0.0
    
    # Use Z-axis (yaw) for horizontal orientation
    yaw_velocities = [gyro['z'] for gyro in gyro_data]
    
    if yaw_velocities:
        # AGGRESSIVE FILTERING to prevent drift
        # 1. Much higher noise threshold
        NOISE_THRESHOLD = 0.15  # Only consider rotations > 0.15 rad/s (much more aggressive)
        yaw_filtered = [v if abs(v) > NOISE_THRESHOLD else 0.0 for v in yaw_velocities]
        
        # 2. Remove outliers (values too extreme)
        yaw_std = np.std(yaw_velocities) if len(yaw_velocities) > 1 else 0
        yaw_mean = np.mean(yaw_velocities)
        outlier_threshold = 2 * yaw_std  # Remove values beyond 2 std deviations
        
        yaw_cleaned = []
        for v in yaw_filtered:
            if v != 0.0 and abs(v - yaw_mean) < outlier_threshold:
                yaw_cleaned.append(v)
            else:
                yaw_cleaned.append(0.0)
        
        # 3. Only use if we have consistent rotation (multiple significant samples)
        significant_samples = sum(1 for v in yaw_cleaned if v != 0.0)
        if significant_samples < len(yaw_cleaned) * 0.3:  # Less than 30% significant = probably noise
            return initial_orientation if initial_orientation is not None else 0.0
        
        # 4. Average filtered angular velocity
        avg_yaw_velocity = np.mean(yaw_cleaned)
        
        # 5. Additional threshold check
        if abs(avg_yaw_velocity) < NOISE_THRESHOLD * 0.5:
            return initial_orientation if initial_orientation is not None else 0.0
        
        # 6. Conservative time integration
        dt = 0.005 * len(yaw_velocities)  # Reduced time step for stability
        
        # 7. Integrate to get orientation change
        delta_orientation = avg_yaw_velocity * dt
        
        # 8. Very conservative delta limits
        MAX_DELTA = 0.05  # rad (~3 degrees per frame - much more conservative)
        delta_orientation = np.clip(delta_orientation, -MAX_DELTA, MAX_DELTA)
        
        # Apply orientation change
        if initial_orientation is not None:
            new_orientation = initial_orientation + delta_orientation
        else:
            new_orientation = delta_orientation
        
        # Normalize to [-π, π]
        new_orientation = np.arctan2(np.sin(new_orientation), np.cos(new_orientation))
        return new_orientation
    
    return initial_orientation if initial_orientation is not None else 0.0


def synchronize_sensors_with_frames(frame_info, sensor_data, window_ms=200):
    """
    Synchronize sensor data with frame data based on timestamps.
    
    Args:
        frame_info (list): List of frame information with timestamps
        sensor_data (dict): Dictionary containing sensor readings
        window_ms (int): Time window in milliseconds for sensor synchronization
        
    Returns:
        list: Synchronized frame-sensor information
    """
    frame_sensor_info = []
    
    for frame in frame_info:
        frame_time = frame['time']
        
        # Find sensor data within temporal window
        accel_window = []
        gyro_window = []
        
        window_seconds = window_ms / 1000
        
        for accel in sensor_data['accelerometer']:
            if abs(accel['time'] - frame_time) <= window_seconds:
                accel_window.append(accel)
        
        for gyro in sensor_data['gyroscope']:
            if abs(gyro['time'] - frame_time) <= window_seconds:
                gyro_window.append(gyro)
        
        # Detect movement type for this frame
        movement_type = detect_movement_type(accel_window, gyro_window)
        
        # Calculate camera orientation
        if len(frame_sensor_info) == 0:
            # First frame - establish initial orientation
            camera_orientation = calculate_camera_orientation(gyro_window, 0.0)
            initial_orientation = camera_orientation
        else:
            # Subsequent frames - use previous orientation as reference
            previous_orientation = frame_sensor_info[-1]['camera_orientation']
            camera_orientation = calculate_camera_orientation(gyro_window, previous_orientation)
        
        frame_sensor_info.append({
            'frame_info': frame,
            'movement_type': movement_type,
            'camera_orientation': camera_orientation,
            'accel_samples': len(accel_window),
            'gyro_samples': len(gyro_window)
        })
    
    # Add initial orientation reference to all frames
    if frame_sensor_info:
        initial_orientation = frame_sensor_info[0]['camera_orientation']
        for info in frame_sensor_info:
            info['initial_orientation'] = initial_orientation
    
    return frame_sensor_info