"""
Extrae frames de un vídeo y los guarda en una carpeta.
"""
import os
import cv2

def extract_frames(video_path, output_folder, every_n=30, ext='jpg'):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n == 0:
            fname = f'frame_{saved_count:05d}.{ext}'
            cv2.imwrite(os.path.join(output_folder, fname), frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f'Frames extraídos: {saved_count} en {output_folder}')
