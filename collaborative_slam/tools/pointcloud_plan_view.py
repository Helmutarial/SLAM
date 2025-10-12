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
    # --- Detección de paredes verticales por variación de Z en celdas X-Y ---
    if 'clean_points' in locals() and 'clean_z' in locals() and clean_points.size > 0:
        grid_size = 0.15  # metros por celda
        x_min, x_max = clean_points[:, 0].min(), clean_points[:, 0].max()
        y_min, y_max = clean_points[:, 1].min(), clean_points[:, 1].max()
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        x_idx = np.digitize(clean_points[:, 0], x_bins) - 1
        y_idx = np.digitize(clean_points[:, 1], y_bins) - 1

        cell_heights = {}
        for xi, yi, zi in zip(x_idx, y_idx, clean_z):
            key = (xi, yi)
            if key not in cell_heights:
                cell_heights[key] = []
            cell_heights[key].append(zi)

        cell_z_ranges = {k: np.ptp(v) for k, v in cell_heights.items() if len(v) > 6}
        if cell_z_ranges:
            z_range_values = np.array(list(cell_z_ranges.values()))
            z_range_threshold = np.percentile(z_range_values, 90)
        else:
            z_range_threshold = 0.5

        for (xi, yi), z_range in cell_z_ranges.items():
            if z_range > z_range_threshold:
                x0 = x_bins[xi]
                y0 = y_bins[yi]
                zs = cell_heights[(xi, yi)]
                z_min, z_max = min(zs), max(zs)
                ax.plot([x0, x0], [y0, y0], c='blue', lw=6, alpha=0.9, marker='|', markersize=18, label='Pared vertical' if 'pared_label' not in locals() else None)
                pared_label = True
    wall_lines = []

    # --- Detección de paredes usando RANSAC (después del filtrado y limpieza) ---
    if 'clean_z' in locals() and clean_z.size > 0:
        from sklearn.linear_model import RANSACRegressor
        from sklearn.cluster import DBSCAN
        min_samples = 80
        residual_threshold = 0.08
        z_floor = np.percentile(clean_z, 10)
        floor_mask = clean_z < z_floor + 0.25
        wall_points = clean_points[floor_mask]

        db = DBSCAN(eps=0.25, min_samples=30).fit(wall_points)
        labels = db.labels_
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster = wall_points[labels == label]
            if len(cluster) < min_samples:
                continue
            ransac = RANSACRegressor(residual_threshold=residual_threshold)
            ransac.fit(cluster[:, 0].reshape(-1, 1), cluster[:, 1])
            x_range = np.linspace(cluster[:, 0].min(), cluster[:, 0].max(), 2)
            y_range = ransac.predict(x_range.reshape(-1, 1))
            wall_lines.append((x_range, y_range))
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
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(10, 8))
    # Puntos en gris tenue como fondo
    scatter = ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c='gray', alpha=0.2, label='Puntos nube')

    # Animación del recorrido
    if trajectory is not None and trajectory.size > 0:
        line, = ax.plot([], [], c='orange', lw=2, label='Trayectoria cámara')
        start = ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Inicio')
        end = ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='Fin')
        def update(num):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=60, blit=True, repeat=False)
        # Flechas de sentido (solo al final para no saturar la animación)
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//20)):
            ax.arrow(trajectory[i, 0], trajectory[i, 1],
                    trajectory[i+1, 0] - trajectory[i, 0],
                    trajectory[i+1, 1] - trajectory[i, 1],
                    shape='full', lw=0, length_includes_head=True,
                    head_width=0.15, head_length=0.3, color='orange', alpha=0.7)
    # Dibujar paredes detectadas (líneas por encima de los puntos)
    for i, (x_range, y_range) in enumerate(wall_lines):
        if i == 0:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95, label='Pared detectada')
        else:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plano 2D acumulado de nubes de puntos + trayectoria + paredes + suelo')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    plt.show()

if __name__ == "__main__":
    main()
