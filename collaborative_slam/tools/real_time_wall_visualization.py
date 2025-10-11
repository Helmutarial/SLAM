"""
Real-time visualization of walls and point clouds.

This script uses SpectacularAI and Open3D to visualize point clouds in real-time
and generate a top-down view of walls as lines.

Requirements: pip install spectacularAI open3d
"""

import spectacularAI
import depthai
import open3d as o3d
import numpy as np
import threading
import time
import os
from utils.file_utils import select_data_folder

class RealTimeWallVisualization:
    def __init__(self, voxelSize=0.05):
        self.voxelSize = voxelSize
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.wall_lines = []

    def run(self):
        print("Close the window to stop visualization")
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

    def updateWalls(self, wall_points):
        """
        Updates the visualization with new wall points.

        Args:
            wall_points (np.ndarray): Points representing walls.
        """
        # Convert wall points to Open3D LineSet
        if len(wall_points) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(wall_points)
            # Create lines connecting consecutive points
            lines = [[i, i + 1] for i in range(len(wall_points) - 1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # Add to visualization
            self.vis.add_geometry(line_set)
            self.wall_lines.append(line_set)

    def clearWalls(self):
        """
        Clears all wall lines from the visualization.
        """
        for line_set in self.wall_lines:
            self.vis.remove_geometry(line_set)
        self.wall_lines = []

def segmentWalls(pointCloud):
    """
    Segments walls from a point cloud using normal orientation.

    Args:
        pointCloud (o3d.geometry.PointCloud): Point cloud.

    Returns:
        np.ndarray: Points representing walls.
    """
    # Estimate normals
    pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Filter points with vertical normals (wall orientation)
    normals = np.asarray(pointCloud.normals)
    verticalIndices = np.where((np.abs(normals[:, 0]) > 0.5) | (np.abs(normals[:, 1]) > 0.5))[0]
    walls = pointCloud.select_by_index(verticalIndices)

    return np.asarray(walls.points)

def main():
    """
    Main entry point of the script.
    """
    # Select data folder
    dataFolder = select_data_folder()

    # Initialize visualization
    visu = RealTimeWallVisualization()

    def onMappingOutput(output):
        for frameId in output.updatedKeyFrames:
            keyFrame = output.map.keyFrames.get(frameId)

            # Check that point cloud exists
            if not keyFrame or not keyFrame.pointCloud:
                continue

            # Segment walls
            pointCloud = o3d.geometry.PointCloud()
            pointCloud.points = o3d.utility.Vector3dVector(keyFrame.pointCloud.getPositionData())
            wall_points = segmentWalls(pointCloud)

            # Update visualization
            visu.clearWalls()
            visu.updateWalls(wall_points)

    print("Starting replay")
    replay = spectacularAI.Replay(dataFolder, onMappingOutput)
    replay.startReplay()

    # Run visualization
    visu.run()
    replay.close()

if __name__ == "__main__":
    main()