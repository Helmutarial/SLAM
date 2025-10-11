import os
import argparse
import sys

def select_data_folder():
    """
    Abre un selector gráfico para elegir la carpeta de grabación.
    Devuelve la ruta seleccionada o sale si no se selecciona nada.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Selecciona la carpeta de grabación (dataFolder)")
        if not folder:
            print("No se seleccionó ninguna carpeta. Saliendo...")
            sys.exit(1)
        return folder
    except Exception as e:
        print(f"Error al abrir el selector de carpetas: {e}")
        sys.exit(1)


def prepare_results_folder(input_folder_path, results_root="results"):
    """
    Crea la estructura de carpetas para guardar los resultados:
    results/<nombre_input_folder>/cloud_points/
    Devuelve la ruta final donde guardar los archivos .ply
    """
    input_folder_name = os.path.basename(os.path.normpath(input_folder_path))
    results_folder = os.path.join(results_root, input_folder_name, "cloud_points")
    os.makedirs(results_folder, exist_ok=True)
    return results_folder


