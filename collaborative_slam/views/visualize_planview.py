import matplotlib.pyplot as plt
from collaborative_slam.utils.constants import CLASS_COLORS as class_colors

def plot_planview(clean_points, detections_3d, trajectory, wall_lines):
    """
    Plotea el plan XY con la nube de puntos, detecciones 3D y trayectoria.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c='gray', alpha=0.2, label='Point cloud')
    cam_positions = None
    if trajectory is not None and trajectory.size > 0:
        cam_positions = trajectory
    # Ploteo de detecciones (todas)
    for det in detections_3d:
        if 'point_3d' in det and det['point_3d'] is not None and 'class' in det:
            color = class_colors.get(det['class'], 'red')
            x, y, _ = det['point_3d']
            conf = det.get('confidence', 0)
            marker = '*' if conf > 0.6 else 'o'
            size = 80 if marker == '*' else 40
            ax.scatter(x, y, s=size, c=color, marker=marker, alpha=0.8, edgecolors='black', linewidths=1.5, label=f"{det['class']} (cloud)" if marker == 'o' else f"{det['class']} (conf>0.5)")
            frame_idx = det.get('frame', None)
            rayo_color = 'green'
            rayo_label = 'Proyección exacta (verde)'
            cam_x, cam_y = None, None
            if cam_positions is not None and frame_idx is not None:
                try:
                    idx = int(frame_idx)
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < len(cam_positions):
                    cam_x, cam_y = cam_positions[idx, 0], cam_positions[idx, 1]
                elif idx is not None and idx >= len(cam_positions):
                    cam_x, cam_y = cam_positions[-1, 0], cam_positions[-1, 1]
                    rayo_color = 'red'
                    rayo_label = 'Proyección anterior (rojo)'
                elif idx is not None and idx < 0:
                    cam_x, cam_y = cam_positions[0, 0], cam_positions[0, 1]
                    rayo_color = 'red'
                    rayo_label = 'Proyección anterior (rojo)'
                else:
                    if cam_positions.shape[0] > 1:
                        cam_x, cam_y = cam_positions[-2, 0], cam_positions[-2, 1]
                        rayo_color = 'red'
                        rayo_label = 'Proyección anterior (rojo)'
            if cam_x is not None and cam_y is not None:
                ax.plot([cam_x, x], [cam_y, y], c=rayo_color, lw=1.5, alpha=0.7, linestyle='--', label=rayo_label)
    # Segunda visualización: solo detecciones con confianza > min_confidence
    min_confidence = 0.5  # Puedes cambiar este valor
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(clean_points[:, 0], clean_points[:, 1], s=2, c='gray', alpha=0.2, label='Point cloud')
    for det in detections_3d:
        if 'point_3d' in det and det['point_3d'] is not None and 'class' in det and det.get('confidence', 0) > min_confidence:
            color = class_colors.get(det['class'], 'red')
            x, y, _ = det['point_3d']
            ax2.scatter(x, y, s=80, c=color, marker='*', alpha=0.8, edgecolors='black', linewidths=1.5, label=f"{det['class']} (conf>{min_confidence})")
            frame_idx = det.get('frame', None)
            rayo_color = 'green'
            rayo_label = 'Proyección exacta (verde)'
            cam_x, cam_y = None, None
            if cam_positions is not None and frame_idx is not None:
                try:
                    idx = int(frame_idx)
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < len(cam_positions):
                    cam_x, cam_y = cam_positions[idx, 0], cam_positions[idx, 1]
                elif idx is not None and idx >= len(cam_positions):
                    cam_x, cam_y = cam_positions[-1, 0], cam_positions[-1, 1]
                    rayo_color = 'red'
                    rayo_label = 'Proyección anterior (rojo)'
                elif idx is not None and idx < 0:
                    cam_x, cam_y = cam_positions[0, 0], cam_positions[0, 1]
                    rayo_color = 'red'
                    rayo_label = 'Proyección anterior (rojo)'
                else:
                    if cam_positions.shape[0] > 1:
                        cam_x, cam_y = cam_positions[-2, 0], cam_positions[-2, 1]
                        rayo_color = 'red'
                        rayo_label = 'Proyección anterior (rojo)'
            if cam_x is not None and cam_y is not None:
                ax2.plot([cam_x, x], [cam_y, y], c=rayo_color, lw=1.5, alpha=0.7, linestyle='--', label=rayo_label)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Detecciones 3D (confianza > {min_confidence})')
    ax2.axis('equal')
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique2 = dict()
    for h, l in zip(handles2, labels2):
        if l not in unique2:
            unique2[l] = h
    ax2.legend(unique2.values(), unique2.keys())
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_facecolor('#f5f5f5')
    plt.show()
    if trajectory is not None and trajectory.size > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], c='orange', lw=2, label='Camera trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=80, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=80, label='End')
    for i, (x_range, y_range) in enumerate(wall_lines):
        ax.plot(x_range, y_range, c='blue', lw=4, alpha=0.95, label='Detected wall' if i == 0 else None)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Detecciones 3D por nube de puntos')
    ax.axis('equal')
    handles, labels = ax.get_legend_handles_labels()
    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys())
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    plt.show()
