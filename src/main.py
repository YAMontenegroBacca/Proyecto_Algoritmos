#1. Configuración de parámetros de entrada para ejecución en cluster
# main.py
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread

#Leer y convertir en matriz dispersa archivo matrix_file
def load_matrix(matrix_file):
    # Lee la matriz dispersa de conteos de expresión génica
    matrix = mmread(matrix_file).tocsc()  # Convertimos a formato columna dispersa (CSC) ya qué la necesitamos así para los siguientes pasos
    return matrix

# Lee los barcodes de cada célula
def load_barcodes(barcodes_file):
    barcodes = pd.read_csv(barcodes_file, header=None, sep='\t')
    barcodes.columns = ['barcode']
    return barcodes

# Lee los identificadores de cada gen
def load_genes(genes_file):
    genes = pd.read_csv(genes_file, header=None, sep='\t')
    genes.columns = ['gene_id', 'gene_name']
    return genes

#explorar los datos para verificar las dimensiones y su integridad
def explore_data(matrix, barcodes, genes):
    print("Dimensiones de la matriz de expresión (genes x células):", matrix.shape)
    print("Número de barcodes (células):", barcodes.shape[0])
    print("Número de genes:", genes.shape[0])

    # Verifica que las dimensiones coincidan
    if matrix.shape[0] == genes.shape[0] and matrix.shape[1] == barcodes.shape[0]:
        print("Las dimensiones coinciden correctamente entre los archivos.")
    else:
        print("Advertencia: las dimensiones no coinciden entre la matriz, barcodes y genes.")
    
    # Muestra algunas filas de genes y barcodes
    print("\nEjemplo de barcodes (células):")
    print(barcodes.head())
    
    print("\nEjemplo de genes:")
    print(genes.head())


#función principal
def main(matrix_file, barcodes_file, genes_file):
    # print(f"Procesando {matrix_file}, {barcodes_file}, {genes_file}")
    # print("Los datos se han cargado")
    
    # Cargar los datos
    matrix = load_matrix(matrix_file)
    barcodes = load_barcodes(barcodes_file)
    genes = load_genes(genes_file)
    
    # Explorar los datos
    explore_data(matrix, barcodes, genes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algoritmo de clustering de single cell")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")

    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file)