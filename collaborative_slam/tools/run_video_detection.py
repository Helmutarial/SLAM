"""
Script interactivo para detección de objetos en vídeo.
- Selecciona el vídeo.
- Extrae frames a results/video/detections/frames
- Detecta objetos y guarda los frames con detecciones en results/video/detections/frames_detected
- Guarda el JSON de detecciones por frame.
"""
import os
import matplotlib.pyplot as plt
import cv2
import json
from collaborative_slam.utils.file_utils import select_video_file
from collaborative_slam.utils.video_utils.extract_frames import extract_frames
from collaborative_slam.utils.yolo_utils.detect_objects import detect_objects_in_folder

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')

def get_parent_folder_name(path):
    return os.path.basename(os.path.dirname(path))

def animate_detections(frames_dir, detections_dir):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    for fname in frame_files:
        frame_path = os.path.join(frames_dir, fname)
        det_path = os.path.join(detections_dir, fname.replace('.jpg', '.json').replace('.png', '.json'))
        img = cv2.imread(frame_path)
        if img is None or not os.path.exists(det_path):
            continue
        with open(det_path, 'r') as f:
            detections = json.load(f)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            conf = det['confidence']
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        plt.clf()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{fname} - Detectados: {[d['class'] for d in detections]}")
        plt.axis('off')
        plt.pause(0.5)
    plt.close()

if __name__ == "__main__":
    video_path = select_video_file()
    if not video_path:
        print('No se seleccionó ningún vídeo.')
        exit(1)
    folder_name = get_parent_folder_name(video_path)
    out_dir = os.path.join(RESULTS_ROOT, folder_name, 'detections')
    frames_dir = os.path.join(out_dir, 'frames')
    detected_dir = os.path.join(out_dir, 'frames_detected')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(detected_dir, exist_ok=True)
    print(f'Extrayendo frames de {video_path}...')
    extract_frames(video_path, frames_dir, every_n=30, ext='jpg')
    print('Detectando objetos en los frames...')
    detect_objects_in_folder(frames_dir, detected_dir)
    print('Animando detecciones...')
    plt.figure(figsize=(8,6))
    animate_detections(frames_dir, detected_dir)
    print('Proceso completado.')
