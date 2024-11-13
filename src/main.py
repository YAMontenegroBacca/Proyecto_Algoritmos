#1. Configuración de parámetros de entrada para ejecución en cluster
# main.py
import argparse

def main(matrix_file, barcodes_file, genes_file):
    # Aquí irá el código del algoritmo
    print(f"Procesando {matrix_file}, {barcodes_file}, {genes_file}")
    print("Los datos se han cargado")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algoritmo de clustering de single cell")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")

    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file)