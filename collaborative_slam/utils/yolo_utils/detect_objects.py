"""
Detecta objetos en frames usando YOLOv8 y guarda las detecciones en JSON.
"""
import os
import json
from ultralytics import YOLO
import cv2

def detect_objects_in_folder(frames_folder, output_folder, model_path='yolov8n.pt', classes_of_interest=None):
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    def get_class_name(class_id):
        names = model.names if hasattr(model, 'names') else model.model.names
        return names[class_id] if class_id < len(names) else str(class_id)
    for fname in sorted(os.listdir(frames_folder)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        frame_path = os.path.join(frames_folder, fname)
        img = cv2.imread(frame_path)
        results = model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = get_class_name(class_id)
                if classes_of_interest and class_name not in classes_of_interest:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'centroid': [cx, cy],
                    'confidence': float(box.conf[0])
                })
        out_path = os.path.join(output_folder, fname.replace('.jpg', '.json').replace('.png', '.json'))
        with open(out_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f'Detecciones guardadas en {out_path}')
