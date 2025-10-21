def test_pointcloud_transform_orders(points, camera_matrix, img_width, img_height, imu_to_cam, cam_to_world):
    print("\n--- TEST: Órdenes de transformación de nube real ---")
    # 1. Solo inversa de camToWorld
    world_to_cam = np.linalg.inv(cam_to_world)
    pts_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam1 = (world_to_cam @ pts_hom.T).T[:, :3]
    pts_mm1 = pts_cam1 * 1000.0
    pts_hom1 = np.hstack([pts_mm1, np.ones((pts_mm1.shape[0], 1))])
    proj1 = (camera_matrix @ pts_hom1[:, :3].T).T
    proj_xy1 = proj1[:, :2] / proj1[:, 2:3]
    depths1 = proj1[:, 2]
    inside1 = np.sum((proj_xy1[:,0] >= 0) & (proj_xy1[:,0] < img_width) & (proj_xy1[:,1] >= 0) & (proj_xy1[:,1] < img_height))
    print(f"Solo inversa camToWorld: dentro={inside1}/{points.shape[0]}, rango XY=({proj_xy1.min():.2f},{proj_xy1.max():.2f}), profundidad=({depths1.min():.2f},{depths1.max():.2f})")

    # 2. Solo imuToCamera
    pts_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam2 = (imu_to_cam @ pts_hom.T).T[:, :3]
    pts_mm2 = pts_cam2 * 1000.0
    pts_hom2 = np.hstack([pts_mm2, np.ones((pts_mm2.shape[0], 1))])
    proj2 = (camera_matrix @ pts_hom2[:, :3].T).T
    proj_xy2 = proj2[:, :2] / proj2[:, 2:3]
    depths2 = proj2[:, 2]
    inside2 = np.sum((proj_xy2[:,0] >= 0) & (proj_xy2[:,0] < img_width) & (proj_xy2[:,1] >= 0) & (proj_xy2[:,1] < img_height))
    print(f"Solo imuToCamera: dentro={inside2}/{points.shape[0]}, rango XY=({proj_xy2.min():.2f},{proj_xy2.max():.2f}), profundidad=({depths2.min():.2f},{depths2.max():.2f})")

    # 3. Primero inversa camToWorld, luego imuToCamera
    pts_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam3 = (world_to_cam @ pts_hom.T).T[:, :3]
    pts_hom3 = np.hstack([pts_cam3, np.ones((pts_cam3.shape[0], 1))])
    pts_cam3b = (imu_to_cam @ pts_hom3.T).T[:, :3]
    pts_mm3 = pts_cam3b * 1000.0
    pts_hom3b = np.hstack([pts_mm3, np.ones((pts_mm3.shape[0], 1))])
    proj3 = (camera_matrix @ pts_hom3b[:, :3].T).T
    proj_xy3 = proj3[:, :2] / proj3[:, 2:3]
    depths3 = proj3[:, 2]
    inside3 = np.sum((proj_xy3[:,0] >= 0) & (proj_xy3[:,0] < img_width) & (proj_xy3[:,1] >= 0) & (proj_xy3[:,1] < img_height))
    print(f"Inversa camToWorld + imuToCamera: dentro={inside3}/{points.shape[0]}, rango XY=({proj_xy3.min():.2f},{proj_xy3.max():.2f}), profundidad=({depths3.min():.2f},{depths3.max():.2f})")
"""
Script to generate depth maps from point clouds and save them as images for each frame.
- Loads point clouds and calibration data
- Projects 3D points to 2D image plane
- For each pixel, stores the minimum depth (distance to camera)
- Saves depth maps as PNG images in results/<video_folder>/depth_maps/
"""
# Imports
import os
import numpy as np
import cv2
import json
from collaborative_slam.utils.file_utils import select_data_folder
from collaborative_slam.utils.pointcloud_utils import load_point_clouds

def load_calibration(calib_path):
    """
    Loads camera calibration matrix and imuToCamera from JSON file.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    cam = calib['cameras'][0]
    camera_matrix = np.array([
        [cam['focalLengthX'], 0, cam['principalPointX']],
        [0, cam['focalLengthY'], cam['principalPointY']],
        [0, 0, 1]
    ])
    img_width = cam['imageWidth']
    img_height = cam['imageHeight']
    imu_to_cam = np.array(cam['imuToCamera'])
    return camera_matrix, img_width, img_height, imu_to_cam

def pointcloud_to_depthmap(points, camera_matrix, img_width, img_height, imu_to_cam, pose):
    """
    Transforms 3D points using only the inverse of camToWorld before projection. Applies x1000 scaling for projection. This fixes the projection so points fall inside the image and have correct depth values.
    """
    # Trabajar todo en metros (sin escalado)
    points_scaled = points
    # Cargar la matriz camToWorld correspondiente al frame
    cam_to_world_path = pose.get('camToWorldPath', None)
    if cam_to_world_path is None:
        raise ValueError("No se encontró la ruta de camToWorld para el frame.")
    with open(cam_to_world_path, 'r') as f:
        cam_to_world = np.array(json.load(f))
    # Calcular la inversa para transformar puntos al sistema de la cámara
    world_to_cam = np.linalg.inv(cam_to_world)
    # Transformar puntos (homogéneos): solo inversa de camToWorld
    pts_hom = np.hstack([points_scaled, np.ones((points_scaled.shape[0], 1))])
    pts_cam = (world_to_cam @ pts_hom.T).T[:, :3]
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    # Project points
    # Escalar a milímetros solo para la proyección
    pts_cam_mm = pts_cam * 1000.0
    pts_hom_cam = np.hstack([pts_cam_mm, np.ones((pts_cam_mm.shape[0], 1))])
    proj = (camera_matrix @ pts_hom_cam[:, :3].T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    depths = proj[:, 2]
    # Diagnóstico de valores de profundidad
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    num_positive = np.sum(depths > 0)
    print(f"Profundidad: min={min_depth:.2f}, max={max_depth:.2f}, positivos={num_positive} de {len(depths)}")
    # Estadísticas de proyección
    min_x, max_x = np.min(proj_xy[:,0]), np.max(proj_xy[:,0])
    min_y, max_y = np.min(proj_xy[:,1]), np.max(proj_xy[:,1])
    print(f"Proyección XY: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    inside = 0
    for i in range(proj_xy.shape[0]):
        x, y = proj_xy[i]
        z = depths[i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img_width and 0 <= yi < img_height:
            inside += 1
            if z > 0 and z < depth_map[yi, xi]:
                depth_map[yi, xi] = z
    print(f"Puntos proyectados dentro de la imagen: {inside} de {proj_xy.shape[0]}")
    # Replace inf with 0 for visualization
    depth_map[depth_map == np.inf] = 0
    return depth_map

def save_depthmap_image(depth_map, out_path):
    """
    Saves depth map as a PNG image (normalized for visualization).
    Now inverts scale: valores bajos = blanco, altos = negro.
    Imprime rango y cobertura.
    """
    norm_map = depth_map.copy()
    valid_pixels = np.count_nonzero(norm_map)
    total_pixels = norm_map.size
    min_val = np.min(norm_map[norm_map > 0]) if valid_pixels > 0 else 0
    max_val = np.max(norm_map)
    print(f"Rango profundidad: min={min_val:.2f}, max={max_val:.2f}, cobertura={(valid_pixels/total_pixels)*100:.2f}%")
    if max_val > 0 and min_val < max_val:
        # Invertir escala: profundidad baja = blanco, alta = negro
        norm_map[norm_map > 0] = 1 - (norm_map[norm_map > 0] - min_val) / (max_val - min_val)
        norm_map = (norm_map * 255).astype(np.uint8)
    else:
        norm_map = norm_map.astype(np.uint8)
    cv2.imwrite(out_path, norm_map)

    # ...existing code...
def test_raw_pointcloud_projection(points, camera_matrix, img_width, img_height):
    print("\n--- TEST: Proyección directa de nube real ---")
    # Proyectar puntos tal cual, asumiendo que están en metros
    pts_mm = points * 1000.0
    pts_hom = np.hstack([pts_mm, np.ones((pts_mm.shape[0], 1))])
    proj = (camera_matrix @ pts_hom[:, :3].T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    depths = proj[:, 2]
    min_x, max_x = np.min(proj_xy[:,0]), np.max(proj_xy[:,0])
    min_y, max_y = np.min(proj_xy[:,1]), np.max(proj_xy[:,1])
    min_depth, max_depth = np.min(depths), np.max(depths)
    print(f"Proyección XY nube real: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    print(f"Profundidad nube real: min={min_depth:.2f}, max={max_depth:.2f}")
    inside = 0
    for i in range(proj_xy.shape[0]):
        x, y = proj_xy[i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img_width and 0 <= yi < img_height:
            inside += 1
    print(f"Puntos reales proyectados dentro de la imagen: {inside} de {points.shape[0]}")

def test_synthetic_projection(camera_matrix, img_width, img_height):
    print("\n--- TEST: Proyección de nube sintética ---")
    # Crear cubo de 1000 puntos entre z=2 y z=3 metros delante de la cámara
    np.random.seed(42)
    cube_points = np.random.uniform([-0.5, -0.5, 2.0], [0.5, 0.5, 3.0], (1000, 3))
    # No aplicar ninguna transformación, solo proyección directa
    pts_mm = cube_points * 1000.0
    pts_hom = np.hstack([pts_mm, np.ones((pts_mm.shape[0], 1))])
    proj = (camera_matrix @ pts_hom[:, :3].T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    depths = proj[:, 2]
    min_x, max_x = np.min(proj_xy[:,0]), np.max(proj_xy[:,0])
    min_y, max_y = np.min(proj_xy[:,1]), np.max(proj_xy[:,1])
    min_depth, max_depth = np.min(depths), np.max(depths)
    print(f"Proyección XY sintética: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    print(f"Profundidad sintética: min={min_depth:.2f}, max={max_depth:.2f}")
    inside = 0
    for i in range(proj_xy.shape[0]):
        x, y = proj_xy[i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img_width and 0 <= yi < img_height:
            inside += 1
    print(f"Puntos sintéticos proyectados dentro de la imagen: {inside} de {cube_points.shape[0]}")
def main():
    # Test de órdenes de transformación con la primera nube real y matrices
    results_folder = select_data_folder()
    cloud_dir = os.path.join(results_folder, 'cloud_points')
    calib_path = os.path.join(results_folder, 'calibration.json')
    depthmap_dir = os.path.join(results_folder, 'depth_maps')
    poses_path = os.path.join(results_folder, 'poses.json')
    os.makedirs(depthmap_dir, exist_ok=True)
    camera_matrix, img_width, img_height, imu_to_cam = load_calibration(calib_path)
    # Ejecutar test sintético antes del pipeline real
    test_synthetic_projection(camera_matrix, img_width, img_height)
    clouds, files = load_point_clouds(cloud_dir)
    # Test de proyección directa con la primera nube real
    if len(clouds) > 0:
        points = np.asarray(clouds[0].points)
        test_raw_pointcloud_projection(points, camera_matrix, img_width, img_height)
        # Test de órdenes de transformación con la primera nube real y matrices
        cam_to_world_path = os.path.join(cloud_dir, f"1_camToWorld.json")
        if os.path.exists(cam_to_world_path):
            with open(cam_to_world_path, 'r') as f:
                cam_to_world = np.array(json.load(f))
            test_pointcloud_transform_orders(points, camera_matrix, img_width, img_height, imu_to_cam, cam_to_world)
    with open(poses_path, 'r') as f:
        poses = {p['frame']: p for p in json.load(f)}
    for cloud, fname in zip(clouds, files):
        points = np.asarray(cloud.points)
        frame_num = int(os.path.splitext(fname)[0])
        pose = poses.get(frame_num, None)
        if pose is None:
            print(f"No pose for frame {frame_num}, skipping...")
            continue
        # Buscar el archivo camToWorld.json correspondiente
        cam_to_world_path = os.path.join(cloud_dir, f"{frame_num}_camToWorld.json")
        if not os.path.exists(cam_to_world_path):
            print(f"No camToWorld para frame {frame_num}, skipping...")
            continue
        pose['camToWorldPath'] = cam_to_world_path
        print(f"Nube {fname}: {points.shape[0]} puntos")
        depth_map = pointcloud_to_depthmap(points, camera_matrix, img_width, img_height, imu_to_cam, pose)
        out_name = os.path.splitext(fname)[0] + '_depth.png'
        out_path = os.path.join(depthmap_dir, out_name)
        save_depthmap_image(depth_map, out_path)
        print(f"Depth map saved: {out_path}")
    print("Todos los depth maps han sido generados y guardados.")

if __name__ == "__main__":
    main()
