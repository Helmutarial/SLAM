"""
Visualiza las detecciones YOLO sobre los frames originales.
"""
import os
import cv2
import json
import matplotlib.pyplot as plt

def visualize_detections(frames_folder, detections_folder):
    for fname in sorted(os.listdir(frames_folder)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        frame_path = os.path.join(frames_folder, fname)
        det_path = os.path.join(detections_folder, fname.replace('.jpg', '.json').replace('.png', '.json'))
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
        plt.figure(figsize=(8,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(fname)
        plt.axis('off')
        plt.show()
        input('Pulsa Enter para ver el siguiente frame...')
