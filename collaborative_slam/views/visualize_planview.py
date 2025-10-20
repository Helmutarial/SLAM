"""
Función para visualizar el plan 2D acumulado de nubes de puntos, trayectoria y detecciones 3D.
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_planview(clean_points, clean_z, clustered_detections, trajectory, class_colors, wall_lines, detections_3d=None):
    """
    Visualiza el plan 2D acumulado de nubes de puntos, trayectoria y detecciones 3D.
    Args:
        clean_points: np.ndarray, puntos filtrados XY
        clean_z: np.ndarray, alturas Z
        clustered_detections: dict, detecciones agrupadas por clase {class_name: [centroid, ...]}
        trajectory: np.ndarray, trayectoria de la cámara
        class_colors: dict, colores por clase
        wall_lines: list, líneas detectadas
        detections_3d: list, detecciones sin clusterizar
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c='gray', alpha=0.2, label='Point cloud')
    # Draw raw detections and projection lines
    if detections_3d:
        for det in detections_3d:
            if 'closest_point' in det and det['closest_point'] is not None and 'class' in det and det.get('pose') is not None:
                color = class_colors.get(det['class'], 'red')
                x, y, _ = det['closest_point']
                cam_pos = det['pose']
                cam_x, cam_y = cam_pos['x'], cam_pos['y']
                # Draw the projection line from camera to closest_point
                ax.plot([cam_x, x], [cam_y, y], c=color, lw=1.5, alpha=0.5, linestyle='--', label=None)
                # Draw the detection point
                ax.scatter(x, y, s=30, c=color, marker='o', alpha=0.7, label=f"{det['class']} (raw)")

    # Trayectoria de la cámara
    if trajectory is not None and trajectory.size > 0:
        line, = ax.plot([], [], c='orange', lw=2, label='Camera trajectory')
        start = ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Start')
        end = ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='End')
        def update(num):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=60, blit=True, repeat=False)
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//20)):
            ax.arrow(trajectory[i, 0], trajectory[i, 1],
                    trajectory[i+1, 0] - trajectory[i, 0],
                    trajectory[i+1, 1] - trajectory[i, 1],
                    shape='full', lw=0, length_includes_head=True,
                    head_width=0.15, head_length=0.3, color='orange', alpha=0.7)

    # Detecciones agrupadas por clase (dibujar al final para que queden por encima)
    if clustered_detections:
        drawn_classes = set()
        count_drawn = 0
        for class_name, centroids in clustered_detections.items():
            color = class_colors.get(class_name, 'red')
            for i, centroid in enumerate(centroids):
                ax.scatter(centroid[0], centroid[1], s=350, c=color, marker='*', edgecolors='black', linewidths=2, zorder=10, label=f'{class_name} (clustered)' if class_name not in drawn_classes else None)
                drawn_classes.add(class_name)
                count_drawn += 1
        print(f"Plotted {count_drawn} clustered detections by class.")
    # Trayectoria de la cámara
    if trajectory is not None and trajectory.size > 0:
        line, = ax.plot([], [], c='orange', lw=2, label='Camera trajectory')
        start = ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Start')
        end = ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='End')
        def update(num):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=60, blit=True, repeat=False)
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//20)):
            ax.arrow(trajectory[i, 0], trajectory[i, 1],
                    trajectory[i+1, 0] - trajectory[i, 0],
                    trajectory[i+1, 1] - trajectory[i, 1],
                    shape='full', lw=0, length_includes_head=True,
                    head_width=0.15, head_length=0.3, color='orange', alpha=0.7)
    # Líneas detectadas (paredes)
    for i, (x_range, y_range) in enumerate(wall_lines):
        if i == 0:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95, label='Detected wall')
        else:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Accumulated 2D plan of point clouds + trajectory + walls + floor + detections')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    try:
        plt.show()
    finally:
        plt.close(fig)
    """
    Visualiza el plan 2D acumulado de nubes de puntos, trayectoria y detecciones 3D.
    Args:
        clean_points: np.ndarray, puntos filtrados XY
        clean_z: np.ndarray, alturas Z
        clustered_detections: dict, detecciones agrupadas por clase {class_name: [centroid, ...]}
        trajectory: np.ndarray, trayectoria de la cámara
        class_colors: dict, colores por clase
        wall_lines: list, líneas detectadas
        min_confidence: float, umbral de confianza para mostrar detecciones
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c='gray', alpha=0.2, label='Point cloud')
    # Detecciones agrupadas por clase
    if clustered_detections:
        drawn_classes = set()
        count_drawn = 0
        for class_name, centroids in clustered_detections.items():
            color = class_colors.get(class_name, 'red')
            for i, centroid in enumerate(centroids):
                ax.scatter(centroid[0], centroid[1], s=220, c=color, marker='*', edgecolors='black', linewidths=2, label=f'{class_name} (clustered)' if class_name not in drawn_classes else None)
                drawn_classes.add(class_name)
                count_drawn += 1
        print(f"Plotted {count_drawn} clustered detections by class.")
    # Trayectoria de la cámara
    if trajectory is not None and trajectory.size > 0:
        line, = ax.plot([], [], c='orange', lw=2, label='Camera trajectory')
        start = ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Start')
        end = ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='End')
        def update(num):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=60, blit=True, repeat=False)
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//20)):
            ax.arrow(trajectory[i, 0], trajectory[i, 1],
                    trajectory[i+1, 0] - trajectory[i, 0],
                    trajectory[i+1, 1] - trajectory[i, 1],
                    shape='full', lw=0, length_includes_head=True,
                    head_width=0.15, head_length=0.3, color='orange', alpha=0.7)
    # Líneas detectadas (paredes)
    for i, (x_range, y_range) in enumerate(wall_lines):
        if i == 0:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95, label='Detected wall')
        else:
            ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Accumulated 2D plan of point clouds + trajectory + walls + floor + detections')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    try:
        plt.show()
    finally:
        plt.close(fig)
