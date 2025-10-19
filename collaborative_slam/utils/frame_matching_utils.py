"""
Utility functions for matching detections to poses based on frame count ratio.
This module provides a function to associate each detection with the closest pose in time,
using the ratio between the number of pose frames and detection frames.
"""

# Imports
from typing import List, Dict


def match_detections_to_poses(poses: List[Dict], detections: List[Dict]) -> List[Dict]:
    """
    Associates each detection with the pose whose 'frame' value matches the calculated target_pose_frame (using ratio).
    If no exact match, returns None for pose and pose_index.
    Args:
        poses (List[Dict]): List of pose dictionaries, each with a 'frame' key.
        detections (List[Dict]): List of detection dictionaries, each with a 'frame' key.
    Returns:
        List[Dict]: List of detections with associated pose index and pose data (or None if not found).
    """
    if not poses or not detections:
        return []
    pose_frames = [p.get('frame', idx) for idx, p in enumerate(poses)]
    last_pose_frame = max(pose_frames)
    last_det_frame = max(d.get('frame', 0) for d in detections)
    if last_det_frame == 0:
        ratio = 1
    else:
        ratio = last_pose_frame / last_det_frame
    matched = []
    for det in detections:
        det_frame = det.get('frame', 0)
        target_pose_frame = int(round(det_frame * ratio))
        # Buscar el índice de pose cuyo frame esté a distancia <= 1 del calculado
        candidates = [(idx, pf) for idx, pf in enumerate(pose_frames) if abs(pf - target_pose_frame) <= 1]
        if candidates:
            # Si hay varios, elige el más cercano
            pose_idx, _ = min(candidates, key=lambda x: abs(x[1] - target_pose_frame))
            pose = poses[pose_idx]
        else:
            pose_idx = None
            pose = None
        detection_with_pose = det.copy()
        detection_with_pose['pose_index'] = pose_idx
        detection_with_pose['pose'] = pose
        matched.append(detection_with_pose)
    return matched

# Recommendation: You can extend this function to support timestamp-based matching if needed.
