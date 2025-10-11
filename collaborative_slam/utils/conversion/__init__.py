"""
Conversion Utilities Module

This module provides tools for converting between different 3D data formats,
specifically for point cloud processing and analysis.

Features:
- PLY to PCD conversion for PCL compatibility
- Format validation and error handling
- Batch processing capabilities
- Optimized binary output formats

Main Functions:
- convert_ply_to_pcd: Convert PLY files to PCD format

Example Usage:
    from utils.conversion import convert_ply_to_pcd
    
    convert_ply_to_pcd("input_folder", "output_folder")
"""

from .convert_pcl import convert_ply_to_pcd, main as run_ply_to_pcd_converter

__version__ = "1.0.0"
__author__ = "TFM Project"
__description__ = "3D data format conversion utilities"

__all__ = [
    'convert_ply_to_pcd',
    'run_ply_to_pcd_converter'
]