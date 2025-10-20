"""
Script maestro para procesamiento de vídeo SLAM: extracción de frames y detección YOLO.
1. Selecciona un vídeo (o carpeta de datos con vídeo).
2. Extrae los frames a la carpeta de resultados correspondiente.
3. Ejecuta YOLO sobre los frames y guarda el JSON de detecciones.
"""
import os
from collaborative_slam.utils.file_utils import select_video_file, create_results_folders
from collaborative_slam.utils.video_utils.extract_frames import extract_frames
from collaborative_slam.utils.yolo_utils.detect_objects import detect_objects_in_folder

def main():
    # Selección de vídeo
    video_path = select_video_file()
    if not video_path:
        print("No se seleccionó ningún vídeo.")
        return
    # Carpeta de resultados: usar la carpeta donde está el vídeo, no el nombre del archivo
    folder_name = os.path.basename(os.path.dirname(video_path))
    results_folder, _ = create_results_folders(folder_name)
    detections_dir = os.path.join(results_folder, 'detections')
    frames_dir = os.path.join(detections_dir, 'frames')
    detected_dir = os.path.join(detections_dir, 'frames_detected')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(detected_dir, exist_ok=True)
    # 1. Extraer frames
    print(f"[INFO] Extrayendo frames de {video_path} ...")
    extract_frames(video_path, frames_dir)
    # 2. Detección YOLO
    print(f"[INFO] Ejecutando YOLO sobre los frames ...")
    detect_objects_in_folder(frames_dir, detected_dir)
    print(f"[INFO] Proceso completado. Frames y detecciones guardados en {detections_dir}")

if __name__ == "__main__":
    main()
