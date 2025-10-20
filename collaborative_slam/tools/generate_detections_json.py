"""
Script para generar el JSON de detecciones a partir de los frames de vídeo usando YOLOv8.
Llama a la función detect_objects_in_folder y guarda el resultado en detections.json.
"""

# Imports
from collaborative_slam.utils.yolo_utils.detect_objects import detect_objects_in_folder
import os

if __name__ == "__main__":
    # Configuración de rutas
    frames_folder = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'detections', 'frames')
    output_folder = os.path.join(os.path.dirname(__file__), '..', 'results', 'VIDEO1', 'detections', 'frames_detected')
    # Llama a la función de detección
    detect_objects_in_folder(frames_folder, output_folder)
    print("JSON de detecciones generado correctamente.")
