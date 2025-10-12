"""
Script para visualizar la trayectoria de la cámara en planta (vista superior, X-Y) usando Matplotlib.
Carga los poses desde poses.json y muestra el recorrido ignorando la altura (Z).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo de poses (ajusta si es necesario)
POSES_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'poses.json'
)

def load_trajectory(poses_path):
    """
    Load camera trajectory from poses.json file.
    Args:
        poses_path (str): Path to poses.json file.
    Returns:
        np.ndarray: Nx3 array with camera positions (x, y, z).
    """
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    trajectory = []
    for pose in poses:
        trajectory.append([pose['x'], pose['y'], pose['z']])
    return np.array(trajectory)

def plot_trajectory_top_view(trajectory):
    """
    Plot camera trajectory from top view (X-Y) using Matplotlib.
    Args:
        trajectory (np.ndarray): Nx3 array with camera positions.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label='Trayectoria (vista superior)')
    # Marcar el inicio (verde) y el final (rojo)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, label='Inicio')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, label='Fin')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trayectoria de la cámara (vista superior)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    trajectory = load_trajectory(POSES_PATH)
    if trajectory.size == 0:
        print("No hay trayectoria para visualizar.")
        return
    plot_trajectory_top_view(trajectory)

if __name__ == "__main__":
    main()
