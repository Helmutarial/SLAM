from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
def load_trajectory(poses_path):
    """
    Carga la trayectoria de la cámara desde un archivo poses.json.
    Args:
        poses_path (str): Ruta al archivo JSON de poses.
    Returns:
        np.ndarray: Array Nx2 con las posiciones X-Y de la trayectoria.
    """
    import json
    if not os.path.exists(poses_path):
        return None
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    return np.array([[p['x'], p['y']] for p in poses])
"""
Script colaborativo para comparar celdas ocupadas y alinear nubes de dos vídeos usando ICP con inicialización automática.
Visualiza:
1. Celdas ocupadas y coincidencias
2. Nubes antes y después de la alineación
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.pointcloud_utils.accumulation import merge_point_clouds
from collaborative_slam.utils.pointcloud_utils.alignment import align_clouds, compute_icp_rmse

def filter_statistical_outlier(cloud, nb_neighbors=20, std_ratio=2.0):
    """
    Filtra el ruido de una nube de puntos usando el método estadístico de Open3D.
    Args:
        cloud (o3d.geometry.PointCloud): Nube de puntos original.
        nb_neighbors (int): Número de vecinos para el análisis.
        std_ratio (float): Umbral de desviación estándar.
    Returns:
        o3d.geometry.PointCloud: Nube filtrada.
    """
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

CLOUD_FOLDER_1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'cloud_points')
CLOUD_FOLDER_2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO2', 'cloud_points')

GRID_SIZE = 0.2  # metros por celda
OCCUPANCY_THRESHOLD = 8  # puntos para considerar celda ocupada


def get_occupancy_grid(points, grid_size, occ_thresh):
    x_min, x_max = np.percentile(points[:, 0], [2, 98])
    y_min, y_max = np.percentile(points[:, 1], [2, 98])
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)
    grid = np.zeros((len(x_bins), len(y_bins)), dtype=int)
    x_idx = np.digitize(points[:, 0], x_bins) - 1
    y_idx = np.digitize(points[:, 1], y_bins) - 1
    for xi, yi in zip(x_idx, y_idx):
        if 0 <= xi < grid.shape[0] and 0 <= yi < grid.shape[1]:
            grid[xi, yi] += 1
    occ_grid = (grid >= occ_thresh).astype(int)
    return occ_grid, x_bins, y_bins


def plot_occupancy_grid(grid, x_bins, y_bins, title):
    plt.figure(figsize=(8, 7))
    plt.imshow(grid.T, origin='lower', cmap='Greys', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.colorbar(label='Ocupación')
    plt.grid(True)
    plt.show()


def main():
    # Cargar trayectorias de ambos vídeos
    poses_path1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'poses.json')
    poses_path2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO2', 'poses.json')
    traj1 = load_trajectory(poses_path1)
    traj2 = load_trajectory(poses_path2)
    clouds1, _ = load_point_clouds(CLOUD_FOLDER_1)
    clouds2, _ = load_point_clouds(CLOUD_FOLDER_2)
    if not clouds1 or not clouds2:
        print('No se encontraron nubes en una de las carpetas.')
        return
    merged1 = merge_point_clouds(clouds1)
    merged2 = merge_point_clouds(clouds2)

    # Filtrar ruido de las nubes antes de cualquier cálculo
    merged1 = filter_statistical_outlier(merged1, nb_neighbors=20, std_ratio=2.0)
    merged2 = filter_statistical_outlier(merged2, nb_neighbors=20, std_ratio=2.0)

    pts1 = np.asarray(merged1.points)[:, :2]
    pts2 = np.asarray(merged2.points)[:, :2]

    # Visualizar nubes originales
    plt.figure(figsize=(10, 8))
    plt.scatter(pts1[:, 0], pts1[:, 1], s=1, c='blue', alpha=0.3, label='Nube 1')
    plt.scatter(pts2[:, 0], pts2[:, 1], s=1, c='red', alpha=0.3, label='Nube 2')
    plt.title('Nubes antes de la alineación')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 1. Alineación robusta con RANSAC+FPFH
    print('Alineando con RANSAC+FPFH...')
    trans_init = align_clouds(merged2, merged1, voxel_size=0.15)
    merged2_aligned = merged2.transform(trans_init)
    pts2_aligned = np.asarray(merged2.points)[:, :2]

    plt.figure(figsize=(10, 8))
    plt.scatter(pts1[:, 0], pts1[:, 1], s=1, c='blue', alpha=0.3, label='Nube 1')
    plt.scatter(pts2_aligned[:, 0], pts2_aligned[:, 1], s=1, c='red', alpha=0.3, label='Nube 2 alineada (RANSAC+FPFH)')
    plt.title('Nubes tras alineación inicial (RANSAC+FPFH)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Refinar con ICP
    print('Refinando con ICP...')
    rmse, trans_icp = compute_icp_rmse(merged2, merged1, trans_init)
    merged2_icp = merged2.transform(trans_icp)
    pts2_icp = np.asarray(merged2.points)[:, :2]
    pts2_icp_z = np.asarray(merged2.points)[:, 2]
    pts1_z = np.asarray(merged1.points)[:, 2]

    # --- Submuestreo y filtrado para visualización ---
    def filter_points(points, z, max_points=8000):
        if points.shape[0] > max_points:
            idx = np.random.choice(points.shape[0], max_points, replace=False)
            return points[idx], z[idx]
        return points, z
    pts1_vis, pts1_z_vis = filter_points(pts1, pts1_z)
    pts2_vis, pts2_z_vis = filter_points(pts2_icp, pts2_icp_z)

    # --- Centrar y limitar rango X/Y ---
    all_xy = np.vstack([pts1_vis, pts2_vis])
    x_min, x_max = np.percentile(all_xy[:, 0], [2, 98])
    y_min, y_max = np.percentile(all_xy[:, 1], [2, 98])
    def mask_range(points):
        return (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    mask1 = mask_range(pts1_vis)
    mask2 = mask_range(pts2_vis)
    pts1_vis, pts1_z_vis = pts1_vis[mask1], pts1_z_vis[mask1]
    pts2_vis, pts2_z_vis = pts2_vis[mask2], pts2_z_vis[mask2]

    # --- Detección de paredes por variación de Z en celdas X-Y ---
    def detect_walls(points, z, grid_size=0.15):
        x_bins = np.arange(points[:, 0].min(), points[:, 0].max() + grid_size, grid_size)
        y_bins = np.arange(points[:, 1].min(), points[:, 1].max() + grid_size, grid_size)
        x_idx = np.digitize(points[:, 0], x_bins) - 1
        y_idx = np.digitize(points[:, 1], y_bins) - 1
        cell_heights = {}
        for xi, yi, zi in zip(x_idx, y_idx, z):
            key = (xi, yi)
            if key not in cell_heights:
                cell_heights[key] = []
            cell_heights[key].append(zi)
        cell_z_ranges = {k: np.ptp(v) for k, v in cell_heights.items() if len(v) > 6}
        wall_lines = []
        if cell_z_ranges:
            z_range_values = np.array(list(cell_z_ranges.values()))
            z_range_threshold = np.percentile(z_range_values, 90)
            for (xi, yi), z_range in cell_z_ranges.items():
                if z_range > z_range_threshold:
                    x0 = x_bins[xi]
                    y0 = y_bins[yi]
                    wall_lines.append((x0, y0))
        return wall_lines
    walls1 = detect_walls(pts1_vis, pts1_z_vis)
    walls2 = detect_walls(pts2_vis, pts2_z_vis)

    # --- Visualización final mejorada ---
    fig, ax = plt.subplots(figsize=(11, 9))
    sc1 = ax.scatter(pts1_vis[:, 0], pts1_vis[:, 1], s=3, c=pts1_z_vis, cmap='Blues', alpha=0.18, label='Nube 1')
    sc2 = ax.scatter(pts2_vis[:, 0], pts2_vis[:, 1], s=3, c=pts2_z_vis, cmap='Reds', alpha=0.18, label='Nube 2 alineada')
    plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02, label='Altura (Z) Nube 1')
    plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04, label='Altura (Z) Nube 2')
    # Dibujar paredes
    for x0, y0 in walls1:
        ax.plot([x0], [y0], c='blue', marker='|', markersize=18, lw=6, alpha=0.9, label='Pared Nube 1' if 'pared1' not in locals() else None)
        pared1 = True
    for x0, y0 in walls2:
        ax.plot([x0], [y0], c='purple', marker='|', markersize=18, lw=6, alpha=0.9, label='Pared Nube 2' if 'pared2' not in locals() else None)
        pared2 = True
    # Trayectorias con flechas y grosor
    if traj1 is not None and traj1.size > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=3, label='Trayectoria 1')
        ax.scatter(traj1[0, 0], traj1[0, 1], color='green', s=80, label='Inicio 1')
        ax.scatter(traj1[-1, 0], traj1[-1, 1], color='red', s=80, label='Fin 1')
        for i in range(0, len(traj1)-1, max(1, len(traj1)//20)):
            ax.arrow(traj1[i, 0], traj1[i, 1], traj1[i+1, 0]-traj1[i, 0], traj1[i+1, 1]-traj1[i, 1], shape='full', lw=0, length_includes_head=True, head_width=0.15, head_length=0.3, color='orange', alpha=0.7)
    if traj2 is not None and traj2.size > 0:
        traj2_h = np.hstack([traj2, np.zeros((traj2.shape[0], 1)), np.ones((traj2.shape[0], 1))])
        traj2_icp = (trans_icp @ traj2_h.T).T[:, :2]
        ax.plot(traj2_icp[:, 0], traj2_icp[:, 1], c='purple', lw=3, label='Trayectoria 2 alineada')
        ax.scatter(traj2_icp[0, 0], traj2_icp[0, 1], color='green', s=80, label='Inicio 2')
        ax.scatter(traj2_icp[-1, 0], traj2_icp[-1, 1], color='red', s=80, label='Fin 2')
        for i in range(0, len(traj2_icp)-1, max(1, len(traj2_icp)//20)):
            ax.arrow(traj2_icp[i, 0], traj2_icp[i, 1], traj2_icp[i+1, 0]-traj2_icp[i, 0], traj2_icp[i+1, 1]-traj2_icp[i, 1], shape='full', lw=0, length_includes_head=True, head_width=0.15, head_length=0.3, color='purple', alpha=0.7)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plano 2D fusionado: nubes, trayectorias y paredes alineadas')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(pts1[:, 0], pts1[:, 1], s=1, c='blue', alpha=0.3, label='Nube 1')
    plt.scatter(pts2_icp[:, 0], pts2_icp[:, 1], s=1, c='red', alpha=0.3, label='Nube 2 alineada (ICP)')
    plt.title(f'Nubes tras alineación colaborativa (ICP) - RMSE: {rmse:.4f}')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Visualización final en planta con trayectorias ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pts1[:, 0], pts1[:, 1], s=2, c='gray', alpha=0.2, label='Nube 1')
    ax.scatter(pts2_icp[:, 0], pts2_icp[:, 1], s=2, c='red', alpha=0.2, label='Nube 2 alineada')
    # Trayectorias
    if traj1 is not None and traj1.size > 0:
        ax.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=2, label='Trayectoria 1')
        ax.scatter(traj1[0, 0], traj1[0, 1], color='green', s=80, label='Inicio 1')
        ax.scatter(traj1[-1, 0], traj1[-1, 1], color='red', s=80, label='Fin 1')
    if traj2 is not None and traj2.size > 0:
        # Transformar la trayectoria 2 igual que la nube 2
        traj2_h = np.hstack([traj2, np.zeros((traj2.shape[0], 1)), np.ones((traj2.shape[0], 1))])
        traj2_icp = (trans_icp @ traj2_h.T).T[:, :2]
        ax.plot(traj2_icp[:, 0], traj2_icp[:, 1], c='purple', lw=2, label='Trayectoria 2 alineada')
        ax.scatter(traj2_icp[0, 0], traj2_icp[0, 1], color='green', s=80, label='Inicio 2')
        ax.scatter(traj2_icp[-1, 0], traj2_icp[-1, 1], color='red', s=80, label='Fin 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plano 2D fusionado: nubes y trayectorias alineadas')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    plt.show()

if __name__ == "__main__":
    main()
