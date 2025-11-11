"""
Planview visualization for point clouds and trajectories (shared for ICP/USIP alignment)

This module provides a function to visualize two point clouds and their trajectories in 2D (plan view),
with flexible coloring and labels. Designed for reuse in both ICP and USIP alignment scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def visualize_planview(
    pts1, pts2, traj1=None, traj2=None, label1='Cloud 1', label2='Cloud 2',
    color1='Blues', color2='Reds', traj_color1='orange', traj_color2='black',
    title='Planview', alpha=0.18, s=3
):
    """
    Visualize two point clouds and their trajectories in 2D (plan view).
    Args:
        pts1 (np.ndarray): Nx3 points for cloud 1
        pts2 (np.ndarray): Mx3 points for cloud 2
        traj1 (np.ndarray): Kx2 or Kx3 trajectory for cloud 1 (optional)
        traj2 (np.ndarray): Lx2 or Lx3 trajectory for cloud 2 (optional)
        label1 (str): Label for cloud 1
        label2 (str): Label for cloud 2
        color1 (str): Matplotlib colormap for cloud 1
        color2 (str): Matplotlib colormap for cloud 2
        traj_color1 (str): Color for trajectory 1
        traj_color2 (str): Color for trajectory 2
        title (str): Plot title
        alpha (float): Alpha for scatter
        s (int): Marker size
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    z1 = pts1[:, 2] if pts1.shape[1] > 2 else np.zeros(len(pts1))
    z2 = pts2[:, 2] if pts2.shape[1] > 2 else np.zeros(len(pts2))
    sc1 = ax.scatter(pts1[:, 0], pts1[:, 1], s=s, c=z1, cmap=color1, alpha=alpha, label=label1)
    sc2 = ax.scatter(pts2[:, 0], pts2[:, 1], s=s, c=z2, cmap=color2, alpha=alpha, label=label2)
    plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label=f'Z height {label1}')
    plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04, label=f'Z height {label2}')
    if traj1 is not None and len(traj1) > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c=traj_color1, lw=3, label=f'Trajectory 1')
    if traj2 is not None and len(traj2) > 0:
        ax.plot(traj2[:, 0], traj2[:, 1], c=traj_color2, lw=3, label=f'Trajectory 2')
    all_xy = np.vstack([pts1[:, :2], pts2[:, :2]])
    x_min, x_max = np.percentile(all_xy[:, 0], [2, 98])
    y_min, y_max = np.percentile(all_xy[:, 1], [2, 98])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()
