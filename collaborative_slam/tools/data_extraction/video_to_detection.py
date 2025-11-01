"""
Script para detección de objetos en vídeo usando YOLO y visualización automática frame a frame.
-
- Busca el vídeo 'rgb_video.mp4' en la carpeta seleccionada.
- Extrae frames y ejecuta YOLO sobre ellos.
- Guarda el JSON de detecciones en results/<nombre_carpeta>/detections/frames_detected/detections.json
- Muestra cada frame con las detecciones dibujadas, avanzando automáticamente cada 500ms.
"""
# Imports
import os
import sys
import cv2
from collaborative_slam.utils.file_utils import select_data_folder
from collaborative_slam.utils.video_utils.extract_frames import extract_n_equally_spaced_frames
from collaborative_slam.utils.yolo_utils.detect_objects import detect_objects_in_folder
import json

def draw_detections_on_frame(frame, detections):
    """
    Dibuja los bounding boxes y clases sobre el frame.
    """
    for det in detections:
        if 'bbox' in det and 'class' in det:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{det['class']} ({det.get('confidence', 0):.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def main():
    # Selección de carpeta de entrada
    input_folder = select_data_folder()
    if not input_folder:
        sys.exit(1)
    video_path = os.path.join(input_folder, "rgb_video.mp4")
    if not os.path.exists(video_path):
        sys.exit(1)
    # Guardar todo en la carpeta de entrada seleccionada
    results_dir = input_folder
    frames_dir = os.path.join(results_dir, 'frames')
    detected_dir = os.path.join(results_dir, 'detections', 'frames_detected')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(detected_dir, exist_ok=True)
    poses_json_path = os.path.join(input_folder, 'poses.json')
    if not os.path.exists(poses_json_path):
        sys.exit(1)
    extract_n_equally_spaced_frames(video_path, frames_dir, poses_json_path=poses_json_path)
    detect_objects_in_folder(frames_dir, detected_dir)
    # Visualización automática frame a frame con detecciones
    detections_json_path = os.path.join(detected_dir, 'detections.json')
    if not os.path.exists(detections_json_path):
        print("No se encontró el archivo de detecciones.")
        return
    with open(detections_json_path, 'r') as f:
        all_detections = json.load(f)
    # Agrupar detecciones por frame
    det_by_frame = {}
    for det in all_detections:
        frame_idx = det.get('frame', None)
        if frame_idx is not None:
            det_by_frame.setdefault(frame_idx, []).append(det)
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        frame_idx = int(''.join(filter(str.isdigit, fname.split('.')[0])))
        frame_path = os.path.join(frames_dir, fname)
        frame = cv2.imread(frame_path)
        detections = det_by_frame.get(frame_idx, [])
        frame_vis = draw_detections_on_frame(frame, detections)
        cv2.imshow('Detecciones', frame_vis)
        key = cv2.waitKey(500)  # 500ms entre frames, ESC para salir
        if key == 27:  # ESC para salir
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
