import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

def load_point_clouds(folder):
    """Carga todas las nubes de puntos de una carpeta y las devuelve en una lista ordenada."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.ply')], key=lambda x: int(x.split('.')[0]))
    clouds = [o3d.io.read_point_cloud(os.path.join(folder, f)) for f in files]
    return clouds, files

def compute_icp_rmse(source, target):
    """Ejecuta ICP y devuelve el error RMSE."""
    threshold = 3  # Umbral para la distancia máxima de correspondencia
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result.inlier_rmse, icp_result.transformation

def compare_point_clouds(folder1, folder2, output_folder):
    """Compara todas las nubes de Ruta 1 con todas las de Ruta 2 usando ICP."""
    clouds1, files1 = load_point_clouds(folder1)
    clouds2, files2 = load_point_clouds(folder2)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (cloud1, file1) in enumerate(zip(clouds1, files1)):
        if i%skip!=0:
            continue
        rmse_values = []
        
        for j, (cloud2, file2) in enumerate(zip(clouds2, files2)):
            if j%skip!=0:
                continue
            rmse, _ = compute_icp_rmse(cloud1, cloud2)
            rmse_values.append(rmse)
            print(f"✅ RMSE entre {file1} y {file2}: {rmse}")
        
        # Guardar gráfico de RMSE
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(rmse_values)), rmse_values, marker='o', linestyle='-', color='b')
        plt.xlabel("Índice de nube en Ruta 2")
        plt.ylabel("RMSE")
        plt.title(f"RMSE de {file1} con todas las nubes de Ruta 2")
        plt.grid()
        graph_path = os.path.join(output_folder, f"rmse_{file1}.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"📊 Gráfico de RMSE guardado en: {graph_path}")

# 📌 Rutas de entrada
folder1 = "outputNW1"
folder2 = "outputNW3"
output_folder = "rmse_output"
skip = 5

def main():
    """Main function for standalone execution."""
    import sys
    
    # Default parameters
    folder1 = "outputNW1"
    folder2 = "outputNW3"
    output_folder = "rmse_output"
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
    if len(sys.argv) >= 4:
        output_folder = sys.argv[3]
    
    print("📊 DETAILED RMSE ANALYZER")
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