import os
import sys
import open3d as o3d

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project utilities
from utils.file_utils import select_data_folder, prepare_results_folder

def convert_ply_to_pcd(input_folder, output_folder):
    """
    Convert PLY files to PCD format for PCL compatibility.
    
    Args:
        input_folder (str): Path to folder containing .ply files
        output_folder (str): Path to output folder for .pcd files
        
    Returns:
        int: Number of files successfully converted
    """
    os.makedirs(output_folder, exist_ok=True)
    converted_count = 0
    
    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]
    
    if not ply_files:
        print("❌ No se encontraron archivos .ply en la carpeta seleccionada.")
        return 0
    
    print(f"📂 Encontrados {len(ply_files)} archivos .ply para convertir...")
    
    for file in sorted(ply_files):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.ply', '.pcd'))
        
        try:
            # Cargar PLY con Open3D
            cloud = o3d.io.read_point_cloud(input_path)
            if not cloud.has_points():
                print(f"⚠️ {file} está vacío o no se pudo cargar.")
                continue
            
            # Guardar como PCD (formato binario para eficiencia)
            success = o3d.io.write_point_cloud(output_path, cloud, write_ascii=False)
            if success:
                print(f"✅ Convertido: {file} -> {os.path.basename(output_path)}")
                converted_count += 1
            else:
                print(f"❌ Error al guardar: {file}")
                
        except Exception as e:
            print(f"❌ Error procesando {file}: {e}")
    
    return converted_count

def main():
    """Main function to run the PLY to PCD converter."""
    print("🔄 CONVERSOR DE PLY A PCD")
    print("="*50)
    print("📋 Este script convierte nubes de puntos:")
    print("   • Entrada: Archivos .ply")
    print("   • Salida: Archivos .pcd (optimizados para PCL)")
    print("   • Ubicación: Carpeta de resultados")
    print()
    
    # Seleccionar carpeta de entrada
    print("📁 Selecciona la carpeta que contiene los archivos .ply:")
    input_folder = select_data_folder()
    
    if not input_folder:
        print("❌ No se seleccionó ninguna carpeta.")
        return 1
    
    folder_name = os.path.basename(input_folder)
    print(f"📂 Carpeta seleccionada: {folder_name}")
    
    # Crear carpeta de resultados
    results_base = prepare_results_folder(input_folder)
    output_folder = os.path.join(os.path.dirname(results_base), "pcd_converted")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"💾 Los archivos .pcd se guardarán en: {output_folder}")
    print()
    
    # Ejecutar conversión
    try:
        converted_count = convert_ply_to_pcd(input_folder, output_folder)
        
        if converted_count > 0:
            print("\n" + "="*50)
            print("✅ CONVERSIÓN COMPLETADA")
            print(f"📊 Archivos convertidos: {converted_count}")
            print(f"📁 Ubicación: {output_folder}")
            print("="*50)
            return 0
        else:
            print("\n❌ No se pudo convertir ningún archivo.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Conversión interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
