import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

def load_point_clouds(folder):
    """Carga todas las nubes de puntos de una carpeta y las devuelve en una lista ordenada."""
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.ply')])
    clouds = [o3d.io.read_point_cloud(f) for f in files]
    return clouds, files

def preprocess_point_cloud(pcd, voxel_size):
    """Reduce el tamaño de la nube de puntos y extrae características FPFH."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Extraer descriptores FPFH
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    return pcd_down, pcd_fpfh

def align_clouds(source, target, voxel_size=5.0):
    """Alinea la nube de puntos usando RANSAC + FPFH y devuelve la transformación inicial."""
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5

    # Alineación con RANSAC basado en características FPFH
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99)
    )

    return result_ransac.transformation

def compute_icp_rmse(source, target, initial_transform):
    """Ejecuta ICP después de la alineación inicial y devuelve el error RMSE."""
    threshold = 2.0  # Ajustar según la precisión deseada

    # Aplicar transformación inicial antes de ICP
    source.transform(initial_transform)

    # Refinamiento con ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return icp_result.inlier_rmse, icp_result.transformation

def compare_point_clouds(folder1, folder2, output_folder):
    """Compara todas las nubes de Ruta 1 con todas las de Ruta 2 y almacena las correspondencias."""
    clouds1, files1 = load_point_clouds(folder1)
    clouds2, files2 = load_point_clouds(folder2)
    
    os.makedirs(output_folder, exist_ok=True)

    best_matches = []  # Almacena la mejor correspondencia (índice de nube de Ruta 2)

    for i, cloud1 in enumerate(clouds1):
        best_rmse = float('inf')
        best_match = -1
        best_transform = np.eye(4)

        for j, cloud2 in enumerate(clouds2):
            initial_transform = align_clouds(cloud1, cloud2)  # Alineación inicial con RANSAC + FPFH
            rmse, refined_transform = compute_icp_rmse(cloud1, cloud2, initial_transform)

            if rmse < best_rmse:
                best_rmse = rmse
                best_match = j
                best_transform = refined_transform  # Guarda la mejor transformación

        best_matches.append(best_match)
        print(f"✅ Nube {i} de Ruta 1 mejor coincide con nube {best_match} de Ruta 2 (RMSE: {best_rmse})")

        # Guardar nube alineada para referencia
        aligned_cloud = clouds1[i].transform(best_transform)
        o3d.io.write_point_cloud(os.path.join(output_folder, f"aligned_{i:07d}.ply"), aligned_cloud)

    # Guardar la correspondencia en un gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(best_matches)), best_matches, marker='o', color='b', label="Mejor coincidencia")
    plt.plot(range(len(best_matches)), best_matches, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel("Índice de nube de puntos en Ruta 1")
    plt.ylabel("Índice de mejor coincidencia en Ruta 2")
    plt.title("Correspondencia entre nubes de puntos alineadas (Ruta 1 vs Ruta 2)")
    plt.legend()
    
    graph_path = os.path.join(output_folder, "correspondencia_alineada.png")
    plt.savefig(graph_path)
    plt.show()
    
    print(f"📊 Gráfico de correspondencia guardado en: {graph_path}")

def main():
    """Main function for standalone execution."""
    import sys
    
    # Default paths
    folder1 = "output_ply3"
    folder2 = "output_ply4"
    output_folder = "correspondencia_output"
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
    if len(sys.argv) >= 4:
        output_folder = sys.argv[3]
    
    print("🔍 POINT CLOUD CORRESPONDENCE ANALYZER")
    print("="*50)
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print(f"Output: {output_folder}")
    
    try:
        compare_point_clouds(folder1, folder2, output_folder)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

