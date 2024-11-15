#1. Configuración de parámetros de entrada para ejecución en cluster
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# --- 1. Leer datos
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


# --- 2. Explorar datos
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

# --- 3. Filtrar 
# Filtrado: Se eliminan genes expresados en pocas células y células con baja cantidad de genes expresados

def filter_data(matrix, barcodes, genes):
    # Verificar que las dimensiones coincidan
    if matrix.shape[0] != genes.shape[0] or matrix.shape[1] != barcodes.shape[0]:
        raise ValueError(
            f"Dimensiones no coinciden: matriz {matrix.shape}, "
            f"barcodes {barcodes.shape[0]}, genes {genes.shape[0]}"
        )
    
    # Umbrales de filtrado
    min_cells_per_gene = 3  # Mínimo número de células que expresan un gen
    min_genes_per_cell = 200  # Mínimo número de genes expresados en una célula
    
    # Filtrar genes: Retener genes que se expresen en al menos min_cells_per_gene células
    gene_counts = (matrix > 0).sum(axis=1).A1  # .A1 convierte matriz a array 1D
    genes_to_keep = gene_counts >= min_cells_per_gene
    filtered_matrix = matrix[genes_to_keep, :]
    filtered_genes = genes.iloc[genes_to_keep]
    
    # Filtrar células: Retener células que expresen al menos min_genes_per_cell genes
    cell_counts = (filtered_matrix > 0).sum(axis=0).A1
    cells_to_keep = cell_counts >= min_genes_per_cell
    filtered_matrix = filtered_matrix[:, cells_to_keep]
    filtered_barcodes = barcodes.iloc[cells_to_keep]
    
    # Imprimir estadísticas del filtrado
    print(f"\nEstadísticas de filtrado:")
    print(f"Genes originales: {matrix.shape[0]}")
    print(f"Genes después del filtrado: {filtered_matrix.shape[0]}")
    print(f"Células originales: {matrix.shape[1]}")
    print(f"Células después del filtrado: {filtered_matrix.shape[1]}")
    
    #Retorna una tupla
    return filtered_matrix, filtered_barcodes, filtered_genes


#---4. Normalizar
def normalize_data(matrix, scale_factor=10000):
    """
    Normaliza la matriz de expresión génica usando normalización por tamaño de biblioteca
    y transformación logarítmica.
    
    Args:
        matrix: matriz dispersa de expresión (genes x células)
        scale_factor: factor de escala para la normalización (default: 10000)
    
    Returns:
        matriz_normalizada: matriz normalizada y transformada
    """
    # 1. Normalización por tamaño de biblioteca
    # Calcular el total de counts por célula
    cell_sums = matrix.sum(axis=0).A1  # Sumar por columnas (células).A1 convierte a array 1D
    
    # Evitar división por cero
    cell_sums = np.where(cell_sums == 0, 1, cell_sums)
    
    # Crear matriz de factores de normalización
    scaling_factors = scale_factor / cell_sums
    
    # Multiplicar cada columna por su factor de normalización
    normalized_matrix = matrix.copy()
    normalized_matrix = normalized_matrix.multiply(scaling_factors[np.newaxis, :])
    
    # 2. Transformación logarítmica log1p (log(x + 1))
    # Convertir a formato COO para operaciones elemento a elemento
    normalized_matrix = normalized_matrix.tocoo()
    normalized_matrix.data = np.log1p(normalized_matrix.data)
    
    # Convertir de vuelta a CSC para eficiencia
    normalized_matrix = normalized_matrix.tocsc()
    
    return normalized_matrix

def calculate_highly_variable_genes(normalized_matrix, n_top_genes=2000):
    """
    Identifica los genes más variables basados en la media y varianza
    de su expresión normalizada.
    
    Args:
        normalized_matrix: matriz normalizada de expresión
        n_top_genes: número de genes variables a seleccionar
    
    Returns:
        highly_variable: índices booleanos de los genes más variables
        stats: DataFrame con estadísticas de los genes
    """
    # Calcular media y varianza para cada gen
    means = np.array(normalized_matrix.mean(axis=1)).flatten()
    vars = np.array(normalized_matrix.power(2).mean(axis=1)).flatten() - means**2
    
    # Crear DataFrame con estadísticas
    stats = pd.DataFrame({
        'mean': means,
        'var': vars,
        'cv2': vars / (means**2 + 1e-6)  # Coeficiente de variación al cuadrado
    })
    
    # Seleccionar los genes más variables
    highly_variable = np.zeros(normalized_matrix.shape[0], dtype=bool)
    highly_variable[np.argsort(stats['cv2'])[-n_top_genes:]] = True
    
    stats['highly_variable'] = highly_variable
    
    return highly_variable, stats

def process_normalized_data(filtered_matrix, filtered_genes, n_top_genes=2000):
    """
    Procesa la matriz filtrada aplicando normalización y selección de genes variables.
    
    Args:
        filtered_matrix: matriz filtrada de expresión
        filtered_genes: DataFrame con información de genes
        n_top_genes: número de genes variables a seleccionar
    
    Returns:
        normalized_matrix: matriz normalizada completa
        hvg_matrix: matriz normalizada solo con genes variables
        hvg_genes: genes seleccionados como altamente variables
        gene_stats: estadísticas de variabilidad de los genes
    """
    # Normalizar datos
    print("Normalizando datos...")
    normalized_matrix = normalize_data(filtered_matrix)
    
    # Identificar genes altamente variables
    print("Identificando genes altamente variables...")
    highly_variable, gene_stats = calculate_highly_variable_genes(
        normalized_matrix, 
        n_top_genes=n_top_genes
    )
    
    # Crear matriz solo con genes variables
    hvg_matrix = normalized_matrix[highly_variable, :]
    hvg_genes = filtered_genes.iloc[highly_variable]
    
    # Añadir estadísticas al DataFrame de genes
    gene_stats.index = filtered_genes.index
    
    print(f"\nEstadísticas de normalización:")
    print(f"Dimensiones matriz normalizada: {normalized_matrix.shape}")
    print(f"Genes altamente variables seleccionados: {highly_variable.sum()}")
    
    return normalized_matrix, hvg_matrix, hvg_genes, gene_stats



#---5. Analisis de componentes principales
def apply_pca(hvg_matrix, n_components=2):#TODO: Definir cuanto componentes, se lo ha dejado a solo 2
    """
    Aplica PCA para reducir la dimensionalidad de los datos.

    Args:
        hvg_matrix: matriz con los genes altamente variables
        n_components: número de componentes principales a mantener
    
    Returns:
        pca_result: matriz con los datos en el espacio reducido de PCA
    """
    # Convertir a matriz densa para PCA
    dense_matrix = hvg_matrix.toarray()
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(dense_matrix.T)  # Transpuesta para que sea células x genes
    
    print(f"PCA completado. Dimensiones del resultado de PCA: {pca_result.shape}")
    return pca_result




#---6. Clustering
def perform_clustering(data, n_clusters=10):
    """
    Aplica KMeans para agrupar las células en función de sus perfiles de expresión génica.

    Args:
        data: matriz de datos reducidos (ej. resultado de PCA)
        n_clusters: número de clusters
    
    Returns:
        labels: etiquetas de cluster para cada célula
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)
    
    print(f"KMeans completado. Número de clusters: {n_clusters}")
    return labels


#--- 7. Visualizacion
def plot_pca(pca_result):
    """Visualiza los resultados de PCA en un gráfico de dispersión."""
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=10, color='blue', alpha=0.6)
    plt.title("Visualización de PCA")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    #plt.show()
    plt.savefig("pca_result.png")

def plot_clusters(pca_result, cluster_labels):
    """Visualiza los clusters en el espacio PCA."""
    plt.figure(figsize=(10, 8))
    unique_clusters = set(cluster_labels)
    colors = plt.cm.get_cmap("rainbow", len(unique_clusters))  # Colores distintos para cada cluster
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], s=10, 
                    color=colors(cluster), label=f'Cluster {cluster}', alpha=0.6)
    
    plt.title("Clustering en el espacio PCA")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    #plt.show()
    plt.savefig("clusters_result.png")





#función principal
def main(matrix_file, barcodes_file, genes_file):

    # Cargar los datos
    matrix = load_matrix(matrix_file)
    barcodes = load_barcodes(barcodes_file)
    genes = load_genes(genes_file)
    
    # Explorar los datos
    explore_data(matrix, barcodes, genes)
    #filtrar datos
    filtered_matrix, filtered_barcodes, filtered_genes = filter_data(matrix, barcodes, genes)

    # Normalizar datos
    # normalized_matrix = normalize_data(filtered_matrix)
    print("\nProcesando y normalizando datos...")
    normalized_matrix, hvg_matrix, hvg_genes, gene_stats = process_normalized_data(
        filtered_matrix, 
        filtered_genes,
        n_top_genes=2000
    )
    
    # Opcional: guardar resultados
    # print("\nGuardando resultados...")
    # mmwrite('normalized_matrix.mtx', normalized_matrix)
    # hvg_genes.to_csv('highly_variable_genes.csv')
    # gene_stats.to_csv('normalizacion/gene_statistics.csv')
    
    print("\nReduciendo dimensionalidad con PCA...")
    pca_result = apply_pca(hvg_matrix)#Envio la matriz solo con los altamente variables
    print("PCA realizado. Visualizando resultados...")
    # Visualización de PCA
    plot_pca(pca_result)

    print("\nRealizando clustering con KMeans...")
    cluster_labels = perform_clustering(pca_result, n_clusters=10)
    print("Clustering realizado. Visualizando clusters...")
     # Visualización de Clustering
    plot_clusters(pca_result, cluster_labels)

    







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algoritmo de clustering de single cell")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")
    #parser.add_argument("--n_clusters", required=True, help="cantidad de clusters a generar")

    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file)