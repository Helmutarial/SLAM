"""
Script para visualizar y comparar las nubes de puntos y trayectorias de dos vídeos en planta (X-Y).
Muestra tres gráficos:
1. Nube y trayectoria del vídeo 1
2. Nube y trayectoria del vídeo 2
3. Ambas nubes y trayectorias combinadas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.pointcloud_utils.accumulation import merge_point_clouds

# Carpetas de nubes y poses
CLOUD_FOLDER_1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'cloud_points')
POSES_PATH_1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'poses.json')
CLOUD_FOLDER_2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO2', 'cloud_points')
POSES_PATH_2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO2', 'poses.json')


def load_trajectory(poses_path):
    import json
    if not os.path.exists(poses_path):
        return None
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    return np.array([[p['x'], p['y']] for p in poses])


def plot_cloud_and_traj(points, traj, color_points, color_traj, title):
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    # Filtrado y centrado igual que en pointcloud_plan_view.py
    x_min, x_max = np.percentile(points[:, 0], [2, 98])
    y_min, y_max = np.percentile(points[:, 1], [2, 98])
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    filtered_points = points[mask]
    # Filtrado de ruido
    if filtered_points.shape[0] > 6:
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(filtered_points)
        distances, _ = nbrs.kneighbors(filtered_points)
        mean_dist = distances[:, 1:].mean(axis=1)
        noise_mask = mean_dist < np.percentile(mean_dist, 90)
        clean_points = filtered_points[noise_mask]
    else:
        clean_points = filtered_points
    plt.figure(figsize=(10, 8))
    plt.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c=color_points, alpha=0.3, label='Puntos nube')
    if traj is not None and traj.size > 0:
        plt.plot(traj[:, 0], traj[:, 1], c=color_traj, lw=2, label='Trayectoria')
        plt.scatter(traj[0, 0], traj[0, 1], color='green', s=80, label='Inicio')
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', s=80, label='Fin')
        # Flechas en la trayectoria
        for i in range(0, len(traj)-1, max(1, len(traj)//20)):
            plt.arrow(traj[i, 0], traj[i, 1],
                      traj[i+1, 0] - traj[i, 0],
                      traj[i+1, 1] - traj[i, 1],
                      shape='full', lw=0, length_includes_head=True,
                      head_width=0.15, head_length=0.3, color=color_traj, alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.axis('equal')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.gca().set_facecolor('#f5f5f5')
    plt.show()


def main():
    # Cargar nubes y trayectorias
    clouds1, _ = load_point_clouds(CLOUD_FOLDER_1)
    clouds2, _ = load_point_clouds(CLOUD_FOLDER_2)
    if not clouds1 or not clouds2:
        print('No se encontraron nubes en una de las carpetas.')
        return
    merged1 = merge_point_clouds(clouds1)
    merged2 = merge_point_clouds(clouds2)
    points1 = np.asarray(merged1.points)[:, :2]
    points2 = np.asarray(merged2.points)[:, :2]
    traj1 = load_trajectory(POSES_PATH_1)
    traj2 = load_trajectory(POSES_PATH_2)

    # Gráfico 1: solo vídeo 1
    plot_cloud_and_traj(points1, traj1, 'blue', 'orange', 'Vídeo 1: nube y trayectoria')
    # Gráfico 2: solo vídeo 2
    plot_cloud_and_traj(points2, traj2, 'red', 'purple', 'Vídeo 2: nube y trayectoria')
    # Gráfico 3: ambos combinados con filtrado y estilo igual
    from sklearn.neighbors import NearestNeighbors
    # Filtrado y centrado para ambas nubes
    all_points = np.vstack([points1, points2])
    x_min, x_max = np.percentile(all_points[:, 0], [2, 98])
    y_min, y_max = np.percentile(all_points[:, 1], [2, 98])
    mask1 = (points1[:, 0] >= x_min) & (points1[:, 0] <= x_max) & (points1[:, 1] >= y_min) & (points1[:, 1] <= y_max)
    mask2 = (points2[:, 0] >= x_min) & (points2[:, 0] <= x_max) & (points2[:, 1] >= y_min) & (points2[:, 1] <= y_max)
    filtered1 = points1[mask1]
    filtered2 = points2[mask2]
    # Filtrado de ruido
    if filtered1.shape[0] > 6:
        nbrs1 = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(filtered1)
        distances1, _ = nbrs1.kneighbors(filtered1)
        mean_dist1 = distances1[:, 1:].mean(axis=1)
        noise_mask1 = mean_dist1 < np.percentile(mean_dist1, 90)
        clean1 = filtered1[noise_mask1]
    else:
        clean1 = filtered1
    if filtered2.shape[0] > 6:
        nbrs2 = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(filtered2)
        distances2, _ = nbrs2.kneighbors(filtered2)
        mean_dist2 = distances2[:, 1:].mean(axis=1)
        noise_mask2 = mean_dist2 < np.percentile(mean_dist2, 90)
        clean2 = filtered2[noise_mask2]
    else:
        clean2 = filtered2
    plt.figure(figsize=(10, 8))
    plt.scatter(clean1[:, 0], clean1[:, 1], s=2, c='blue', alpha=0.3, label='Nube 1')
    plt.scatter(clean2[:, 0], clean2[:, 1], s=2, c='red', alpha=0.3, label='Nube 2')
    if traj1 is not None and traj1.size > 0:
        plt.plot(traj1[:, 0], traj1[:, 1], c='orange', lw=2, label='Trayectoria 1')
        plt.scatter(traj1[0, 0], traj1[0, 1], color='green', s=80, label='Inicio 1')
        plt.scatter(traj1[-1, 0], traj1[-1, 1], color='red', s=80, label='Fin 1')
        for i in range(0, len(traj1)-1, max(1, len(traj1)//20)):
            plt.arrow(traj1[i, 0], traj1[i, 1],
                      traj1[i+1, 0] - traj1[i, 0],
                      traj1[i+1, 1] - traj1[i, 1],
                      shape='full', lw=0, length_includes_head=True,
                      head_width=0.15, head_length=0.3, color='orange', alpha=0.7)
    if traj2 is not None and traj2.size > 0:
        plt.plot(traj2[:, 0], traj2[:, 1], c='purple', lw=2, label='Trayectoria 2')
        plt.scatter(traj2[0, 0], traj2[0, 1], color='lime', s=80, label='Inicio 2')
        plt.scatter(traj2[-1, 0], traj2[-1, 1], color='magenta', s=80, label='Fin 2')
        for i in range(0, len(traj2)-1, max(1, len(traj2)//20)):
            plt.arrow(traj2[i, 0], traj2[i, 1],
                      traj2[i+1, 0] - traj2[i, 0],
                      traj2[i+1, 1] - traj2[i, 1],
                      shape='full', lw=0, length_includes_head=True,
                      head_width=0.15, head_length=0.3, color='purple', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparativa: nubes y trayectorias de ambos vídeos')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.gca().set_facecolor('#f5f5f5')
    plt.show()

if __name__ == "__main__":
    main()
