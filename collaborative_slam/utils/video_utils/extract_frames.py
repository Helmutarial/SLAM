# Extrae todos los frames desde 0 hasta el valor máximo de 'frame' en poses.json.
import os
import cv2
import json
import numpy as np  # Fix: numpy import


def get_max_frame_from_poses(poses_json_path):
    """
    Returns the maximum frame index from a poses.json file.
    """
    import json
    if not os.path.exists(poses_json_path):
        raise FileNotFoundError(f"poses.json not found at {poses_json_path}")
    with open(poses_json_path, 'r') as f:
        poses = json.load(f)
    return max(p['frame'] for p in poses)

def extract_n_equally_spaced_frames(video_path, output_dir, N=None, poses_json_path=None, results_root=None):
    """
    Extracts N equally spaced frames from a video. If N is None and poses_json_path is provided,
    N will be set to the max frame in poses.json + 1.
    Permite especificar results_root para guardar los frames en una ruta personalizada.
    """
    import cv2
    import os
    if results_root:
        output_dir = os.path.join(results_root, os.path.basename(output_dir))
    if N is None and poses_json_path:
        N = get_max_frame_from_poses(poses_json_path) + 1
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Determinar extensión del video
    ext = 'jpg'  # Forzar extensión de imagen
    # Calcula los índices equiespaciados, redondeando siempre hacia arriba
    if N is None:
        N = total_frames
    raw_indices = np.linspace(0, total_frames - 1, N)
    indices = np.ceil(raw_indices).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)  # Asegura que no se salga del rango
    saved_count = 0
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Quitar debug
        if not ret:
            print(f"[WARNING] Could not read frame {frame_idx}")
            continue
        fname = f'frame_{idx:05d}.{ext}'
        cv2.imwrite(os.path.join(output_dir, fname), frame)
        saved_count += 1
    cap.release()
    print(f'Frames extracted: {saved_count} (equally spaced from 0 to {total_frames - 1}) in {output_dir}')