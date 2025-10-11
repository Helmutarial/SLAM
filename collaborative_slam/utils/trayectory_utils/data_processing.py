"""
Data Processing Module

This module handles the loading, parsing, and initial processing of JSONL data files
containing visual and sensor information.
"""

import json
import numpy as np


def load_jsonl_data(jsonl_path):
    """
    Load and parse JSONL data file.
    
    Args:
        jsonl_path (str): Path to the JSONL data file
        
    Returns:
        list: Parsed entries from the JSONL file
        
    Raises:
        FileNotFoundError: If the JSONL file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
            entries = [json.loads(line.strip()) for line in lines if line.strip()]
        
        print(f"📄 Loaded {len(entries)} entries from {jsonl_path}")
        return entries
        
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {jsonl_path}: {e}")


def extract_sensor_data(entries):
    """
    Extract sensor data (accelerometer and gyroscope) from JSONL entries.
    
    Args:
        entries (list): Parsed JSONL entries
        
    Returns:
        dict: Dictionary with 'accelerometer' and 'gyroscope' data lists
    """
    sensor_data = {'accelerometer': [], 'gyroscope': []}
    
    for entry in entries:
        if "sensor" in entry:
            sensor_type = entry["sensor"].get("type")
            values = entry["sensor"].get("values", [])
            timestamp = entry.get("time", 0)
            
            if sensor_type in sensor_data and len(values) >= 3:
                sensor_data[sensor_type].append({
                    'time': timestamp,
                    'x': values[0],
                    'y': values[1],
                    'z': values[2],
                    'magnitude': np.linalg.norm(values)
                })
    
    return sensor_data


def print_data_summary(raw_poses, frame_info, sensor_data):
    """
    Print a summary of the extracted data.
    
    Args:
        raw_poses (list): List of raw pose data
        frame_info (list): List of frame information
        sensor_data (dict): Dictionary containing sensor data
    """
    print(f"📊 Data extraction summary:")
    print(f"   • Visual poses: {len(raw_poses)}")
    print(f"   • Frame info entries: {len(frame_info)}")
    print(f"   • Accelerometer readings: {len(sensor_data['accelerometer'])}")
    print(f"   • Gyroscope readings: {len(sensor_data['gyroscope'])}")
    
    # Time range analysis
    if frame_info:
        times = [frame['time'] for frame in frame_info]
        print(f"   • Time range: {min(times):.1f}s to {max(times):.1f}s")
        print(f"   • Duration: {max(times) - min(times):.1f}s")


def validate_data_integrity(raw_poses, frame_info, sensor_data):
    """
    Validate the integrity and consistency of the extracted data.
    
    Args:
        raw_poses (list): List of raw pose data
        frame_info (list): List of frame information
        sensor_data (dict): Dictionary containing sensor data
        
    Returns:
        dict: Validation results with warnings and errors
    """
    validation = {
        'errors': [],
        'warnings': [],
        'is_valid': True
    }
    
    # Check if we have pose data
    if not raw_poses:
        validation['errors'].append("No visual poses found")
        validation['is_valid'] = False
    
    # Check if pose and frame info match
    if len(raw_poses) != len(frame_info):
        validation['warnings'].append(f"Pose count ({len(raw_poses)}) doesn't match frame info count ({len(frame_info)})")
    
    # Check sensor data availability
    for sensor_type, data in sensor_data.items():
        if not data:
            validation['warnings'].append(f"No {sensor_type} data found")
        elif len(data) < 100:
            validation['warnings'].append(f"Very few {sensor_type} readings ({len(data)})")
    
    # Check time consistency
    if frame_info:
        times = [frame['time'] for frame in frame_info]
        if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
            validation['warnings'].append("Frame timestamps are not in chronological order")
    
    # Print validation results
    if validation['errors']:
        print("❌ Data validation errors:")
        for error in validation['errors']:
            print(f"   • {error}")
    
    if validation['warnings']:
        print("⚠️ Data validation warnings:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    if validation['is_valid'] and not validation['warnings']:
        print("✅ Data validation passed")
    
    return validation


def save_processed_data(processed_poses, output_path):
    """
    Save processed pose data to JSON file.
    
    Args:
        processed_poses (list): List of processed pose data
        output_path (str): Path where to save the JSON file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(output_path, "w") as f:
            json.dump(processed_poses, f, indent=2)
        
        print(f"💾 Saved {len(processed_poses)} processed poses to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving data to {output_path}: {e}")
        return False


def calculate_data_statistics(poses):
    """
    Calculate statistics for the processed pose data.
    
    Args:
        poses (list): List of processed poses
        
    Returns:
        dict: Dictionary containing various statistics
    """
    if not poses:
        return {}
    
    # Extract positions and times
    positions = np.array([[p["x"], p["y"]] for p in poses])
    times = [p.get("time", 0) for p in poses]
    
    # Calculate movement statistics
    total_distance = 0
    for i in range(1, len(positions)):
        total_distance += np.linalg.norm(positions[i] - positions[i-1])
    
    # Calculate bounding box
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # Time statistics
    duration = max(times) - min(times) if times else 0
    
    stats = {
        'total_poses': len(poses),
        'total_distance': total_distance,
        'bounding_box': {
            'x_range': x_max - x_min,
            'y_range': y_max - y_min,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        },
        'time_stats': {
            'start_time': min(times) if times else 0,
            'end_time': max(times) if times else 0,
            'duration': duration,
            'average_fps': len(poses) / duration if duration > 0 else 0
        }
    }
    
    return stats