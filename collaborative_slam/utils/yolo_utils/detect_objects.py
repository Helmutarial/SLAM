"""
Detecta objetos en frames usando YOLOv8 y guarda las detecciones en JSON.
"""
import os
import json
from ultralytics import YOLO
import cv2

def detect_objects_in_folder(frames_folder, output_folder, model_path='yolov8n.pt', classes_of_interest=None):
    """
    Detecta solo objetos estáticos, excluyendo personas y objetos móviles.
    """
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    # Lista de clases móviles a excluir
    mobile_classes = set([
        'person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'car', 'bicycle', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'airplane', 'skateboard', 'surfboard',
        'sports ball', 'frisbee', 'kite', 'baseball bat', 'baseball glove', 'tennis racket', 'snowboard', 'suitcase',
        'backpack', 'handbag', 'umbrella', 'wheelchair', 'stroller', 'bench', 'toilet', 'remote', 'cell phone', 'mouse', 'keyboard'
    ])
    def get_class_name(class_id):
        names = model.names if hasattr(model, 'names') else model.model.names
        return names[class_id] if class_id < len(names) else str(class_id)
    all_detections = []
    for fname in sorted(os.listdir(frames_folder)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        frame_path = os.path.join(frames_folder, fname)
        img = cv2.imread(frame_path)
        results = model(img)
        frame_number = int(''.join(filter(str.isdigit, fname.split('.')[0])))
        frame_detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = get_class_name(class_id)
                # Filtrar solo clases estáticas
                if class_name in mobile_classes:
                    continue
                if classes_of_interest and class_name not in classes_of_interest:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                frame_detections.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'centroid': [cx, cy],
                    'confidence': float(box.conf[0]),
                    'frame': frame_number
                })
        if frame_detections:
            all_detections.extend(frame_detections)
        else:
            # Si no hay detecciones, guardar objeto vacío con el id del frame
            all_detections.append({'frame': frame_number})
    out_path = os.path.join(output_folder, 'detections.json')
    with open(out_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f'Todas las detecciones guardadas en {out_path}')
