"""
Enum for status of a point cloud or keyframe in the visualization.
"""

from enum import Enum

class Status(Enum):
    """
    Status of a point cloud or keyframe in the visualization.
    """
    VALID = 0
    NEW = 1
    UPDATED = 2
    REMOVED = 3
