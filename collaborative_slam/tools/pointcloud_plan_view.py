"""
Script para visualizar el plano 2D acumulado de todas las nubes de puntos generadas por SLAM.
Fusiona todas las nubes guardadas en cloud_points y proyecta los puntos al plano X-Y.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from collaborative_slam.utils.pointcloud_utils import load_point_clouds
from collaborative_slam.utils.pointcloud_utils.accumulation import merge_point_clouds

# Carpeta donde se guardan las nubes de puntos (ajusta si es necesario)
CLOUD_POINTS_FOLDER = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'cloud_points'
)

def main():
    print(f"Cargando nubes de puntos desde: {CLOUD_POINTS_FOLDER}")
    clouds, files = load_point_clouds(CLOUD_POINTS_FOLDER)
    if not clouds:
        print("No se encontraron nubes de puntos.")
        return
    print(f"Fusionando {len(clouds)} nubes...")
    merged_cloud = merge_point_clouds(clouds)
    points = np.asarray(merged_cloud.points)
    if points.size == 0:
        print("La nube fusionada está vacía.")
        return
    # Proyección al plano X-Y y obtención de Z
    points_xy = points[:, :2]
    points_z = points[:, 2]

    # --- Filtro: limitar rango X/Y para centrar la visualización ---
    x_min, x_max = np.percentile(points_xy[:, 0], [2, 98])
    y_min, y_max = np.percentile(points_xy[:, 1], [2, 98])
    mask = (points_xy[:, 0] >= x_min) & (points_xy[:, 0] <= x_max) & (points_xy[:, 1] >= y_min) & (points_xy[:, 1] <= y_max)
    filtered_points = points_xy[mask]
    filtered_z = points_z[mask]

    # --- Filtrado de ruido: eliminar puntos aislados ---
    # Usamos un filtro simple por distancia a los vecinos (puedes ajustar el umbral)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(filtered_points)
    distances, _ = nbrs.kneighbors(filtered_points)
    mean_dist = distances[:, 1:].mean(axis=1)
    noise_mask = mean_dist < np.percentile(mean_dist, 90)  # Elimina el 10% más aislado
    clean_points = filtered_points[noise_mask]
    clean_z = filtered_z[noise_mask]

    # --- Cargar y dibujar la trayectoria de la cámara ---
    poses_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'poses.json')
    if os.path.exists(poses_path):
        import json
        with open(poses_path, 'r') as f:
            poses = json.load(f)
        trajectory = np.array([[p['x'], p['y']] for p in poses])
    else:
        trajectory = None

    # --- Visualización: colorear por altura (Z) ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c=clean_z, cmap='viridis', alpha=0.4, label='Puntos nube')
    plt.colorbar(scatter, label='Altura (Z)')
    if trajectory is not None and trajectory.size > 0:
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='orange', lw=2, label='Trayectoria cámara')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Inicio')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='Fin')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plano 2D acumulado de nubes de puntos + trayectoria (color Z)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
