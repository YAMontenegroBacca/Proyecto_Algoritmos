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
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score

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
    min_cells_per_gene = 10  # Mínimo número de células que expresan un gen
    min_genes_per_cell = 300  # Mínimo número de genes expresados en una célula
    
    

    # Filtrar genes: Retener genes que se expresen en al menos min_cells_per_gene células
    gene_counts = (matrix > 0).sum(axis=1).A1  # .A1 convierte matriz a array 1D
    genes_below_threshold = np.sum(gene_counts < min_cells_per_gene)
    genes_to_keep = gene_counts >= min_cells_per_gene
    filtered_matrix = matrix[genes_to_keep, :]
    filtered_genes = genes.iloc[genes_to_keep]
    
    # Filtrar células: Retener células que expresen al menos min_genes_per_cell genes
    cell_counts = (filtered_matrix > 0).sum(axis=0).A1
    cells_below_threshold = np.sum(cell_counts < min_genes_per_cell)
    cells_to_keep = cell_counts >= min_genes_per_cell
    filtered_matrix = filtered_matrix[:, cells_to_keep]
    filtered_barcodes = barcodes.iloc[cells_to_keep]

    # Mostrar información en consola
    print(f"\nMínimo número de células que expresan un gen: {min_cells_per_gene}")
    print(f"Genes por debajo de este umbral: {genes_below_threshold}")
    print(f"Mínimo número de genes expresados por célula: {min_genes_per_cell}")
    print(f"Células por debajo de este umbral: {cells_below_threshold}")
    
    # Imprimir estadísticas del filtrado
    print(f"\nEstadísticas de filtrado:")
    print(f"Genes originales: {matrix.shape[0]}")
    print(f"Genes después del filtrado: {filtered_matrix.shape[0]}")
    print(f"Células originales: {matrix.shape[1]}")
    print(f"Células después del filtrado: {filtered_matrix.shape[1]}")
    
    #Retorna una tupla
    return filtered_matrix, filtered_barcodes, filtered_genes


# def filter_interactively(matrix, barcodes, genes):
#     while True:
#         # Valores predeterminados
#         min_cells_per_gene = 3
#         min_genes_per_cell = 200

#         # Filtra datos y calcula estadísticas
#         gene_counts = (matrix > 0).sum(axis=1).A1
#         cell_counts = (matrix > 0).sum(axis=0).A1

#         genes_below_threshold = np.sum(gene_counts < min_cells_per_gene)
#         cells_below_threshold = np.sum(cell_counts < min_genes_per_cell)

#         # Crear gráfica
#         plt.figure(figsize=(10, 6))
#         plt.hist(gene_counts, bins=50, alpha=0.7, label="Genes")
#         plt.axvline(min_cells_per_gene, color='r', linestyle='--', label="Umbral Genes")
#         plt.hist(cell_counts, bins=50, alpha=0.7, label="Células")
#         plt.axvline(min_genes_per_cell, color='b', linestyle='--', label="Umbral Células")
#         plt.legend()
#         plt.title("Distribución de genes y células filtradas")
#         plt.xlabel("Expresión")
#         plt.ylabel("Frecuencia")
#         plt.show()

#         # Mostrar información en consola
#         print(f"\nMínimo número de células que expresan un gen: {min_cells_per_gene}")
#         print(f"Genes por debajo de este umbral: {genes_below_threshold}")
#         print(f"Mínimo número de genes expresados por célula: {min_genes_per_cell}")
#         print(f"Células por debajo de este umbral: {cells_below_threshold}")

#         # Solicitar confirmación o nuevos valores
#         user_input = input("¿Desea continuar con estos parámetros? (s/n): ").lower()
#         if user_input == 's':
#             break
#         else:
#             min_cells_per_gene = int(input("Ingrese un nuevo mínimo de células por gen: "))
#             min_genes_per_cell = int(input("Ingrese un nuevo mínimo de genes por célula: "))
    
#     # Filtra usando los valores finales
#     return filter_data(matrix, barcodes, genes, min_cells_per_gene, min_genes_per_cell)






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

def calculate_highly_variable_genes(normalized_matrix, n_top_genes=500):
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

def select_highly_variable_genes_interactively(normalized_matrix, filtered_genes):
    while True:
        # Sugerencia inicial basada en el percentil 85
        variances = np.array(normalized_matrix.power(2).mean(axis=1)).flatten() - \
                    np.array(normalized_matrix.mean(axis=1)).flatten()**2
        suggested_genes = np.sum(variances > np.percentile(variances, 85))

        print(f"\nCantidad sugerida de genes altamente variables: {suggested_genes}")
        n_top_genes = int(input("Ingrese la cantidad de genes a seleccionar (sugerido: {}): ".format(suggested_genes)))

        highly_variable, stats = calculate_highly_variable_genes(normalized_matrix, n_top_genes=n_top_genes)

        # Gráfica
        plt.figure(figsize=(10, 6))
        plt.scatter(stats['mean'], stats['var'], alpha=0.5, label='Genes')
        plt.scatter(stats.loc[highly_variable, 'mean'], stats.loc[highly_variable, 'var'], color='red', label='Genes Seleccionados')
        plt.title("Selección de Genes Altamente Variables")
        plt.xlabel("Media")
        plt.ylabel("Varianza")
        plt.legend()
        plt.show()

        # Mostrar resumen estadístico
        print(stats.describe())

        user_input = input("¿Desea continuar con esta cantidad de genes? (s/n): ").lower()
        if user_input == 's':
            break

    return highly_variable, stats







def process_normalized_data(filtered_matrix, filtered_genes):
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
    highly_variable, gene_stats = select_highly_variable_genes_interactively(normalized_matrix, filtered_genes)
    # highly_variable, gene_stats = calculate_highly_variable_genes(
    #     normalized_matrix, 
    #     n_top_genes=n_top_genes
    # )
    
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

#Determina la cnatidad de componentes principales
def determine_pca_dimensions_interactively(hvg_matrix):
    dense_matrix = hvg_matrix.toarray()
    pca = PCA()
    pca.fit(dense_matrix.T)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(0.85, color='r', linestyle='--', label='85% varianza explicada')
    plt.title("Varianza explicada acumulada")
    plt.xlabel("Número de componentes principales")
    plt.ylabel("Varianza acumulada")
    plt.legend()
    plt.show()

    suggested_dimensions = np.argmax(cumulative_variance >= 0.85) + 1
    print(f"Sugerencia: {suggested_dimensions} componentes principales")

    n_components = int(input("Ingrese el número de componentes principales a usar (sugerido: {}): ".format(suggested_dimensions)))

    return n_components

def apply_pca(hvg_matrix):
    """
    Aplica PCA para reducir la dimensionalidad de los datos.

    Args:
        hvg_matrix: matriz con los genes altamente variables
        n_components: número de componentes principales a mantener
    
    Returns:
        pca_result: matriz con los datos en el espacio reducido de PCA
    """
    #Calcular o aceptar cantidad de componentes de usuario
    n_components = determine_pca_dimensions_interactively(hvg_matrix)
    # Convertir a matriz densa para PCA
    dense_matrix = hvg_matrix.toarray()
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(dense_matrix.T)  # Transpuesta para que sea células x genes
    
    print(f"PCA completado. Dimensiones del resultado de PCA: {pca_result.shape}")
    return pca_result






#---6. Clustering
def perform_clustering(data, n_clusters=2):
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

def pearson_distance_matrix(matrix, centers):
    """
    Calcula una matriz de distancia basada en la correlación de Pearson
    entre cada punto en `matrix` y cada punto en `centers`.
    """
    # Normalizar filas de ambas matrices
    matrix_normalized = (matrix - np.mean(matrix, axis=1, keepdims=True)) / np.std(matrix, axis=1, keepdims=True)
    centers_normalized = (centers - np.mean(centers, axis=1, keepdims=True)) / np.std(centers, axis=1, keepdims=True)
    
    # Producto punto para correlaciones
    correlation_matrix = np.dot(matrix_normalized, centers_normalized.T)
    
    # Convertir correlación a distancia
    distance_matrix = 1 - correlation_matrix
    return distance_matrix


def kmeans_with_pearson(pca_result, n_clusters, max_iter=300, tol=1e-4):
    """
    KMeans modificado usando distancia basada en correlación de Pearson.
    """
    # Inicialización aleatoria de centroides
    indices = np.random.choice(pca_result.shape[0], n_clusters, replace=False)
    centers = pca_result[indices]

    for iteration in range(max_iter):
        # Calcula la distancia Pearson entre cada punto y los centroides
        distances = np.zeros((pca_result.shape[0], n_clusters))
        for k in range(n_clusters):
            # Calcula las distancias de Pearson entre todas las células y el centroide k
            center_reshaped = centers[k].reshape(1, -1)  # Aseguramos que el centroide sea una matriz 2D
            distances[:, k] = pearson_distance_matrix(pca_result, center_reshaped).flatten()

        # Asigna cada punto al cluster más cercano
        labels = np.argmin(distances, axis=1)

        # Recalcula los centroides como el promedio de los puntos asignados
        new_centers = np.array([pca_result[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Verifica convergencia
        if np.allclose(centers, new_centers, atol=tol):
            print(f"Convergencia alcanzada en iteración {iteration + 1}")
            break
        centers = new_centers

    return labels, centers


#--- 7. Modularidad
def label_propagation(matrix, n_neighbors=10, max_iter=100):
    """
    Algoritmo de propagación de etiquetas para detectar comunidades.
    """
    # Crear un grafo de vecinos cercanos
    graph = kneighbors_graph(matrix, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    labels = np.arange(matrix.shape[0])  # Etiquetas iniciales únicas para cada célula
    
    for _ in range(max_iter):
        for i in range(matrix.shape[0]):
            # Vecinos de la célula actual
            neighbors = graph[i].indices
            # Asignar la etiqueta más frecuente de los vecinos
            labels[i] = np.bincount(labels[neighbors]).argmax()
        
        # Verificar convergencia
        unique_labels = np.unique(labels)
        if len(unique_labels) == len(np.unique(labels)):
            break
    
    return labels




#--- 8. Visualizacion
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



#--- 9. Guardar resultados
def save_clustering_results(pca_result, cluster_labels, output_file="clustering_results.csv"):
    """
    Guarda los resultados del clustering en un archivo CSV.
    """
    results = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': cluster_labels
    })
    results.to_csv(output_file, index=False)
    print(f"Resultados de clustering guardados en {output_file}")


def calculate_silhouette_score(pca_result, cluster_labels):
    """
    Calcula el Silhouette Score para evaluar la calidad del clustering.
    """
    score = silhouette_score(pca_result, cluster_labels)
    print(f"Silhouette Score: {score:.4f}")
    return score


def plot_clusters_with_labels(pca_result, cluster_labels):
    plt.figure(figsize=(10, 8))
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=f'Cluster {cluster}', alpha=0.6)
    plt.title("Clustering en el espacio PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig("clustering_with_labels.png")




#función principal
def main(matrix_file, barcodes_file, genes_file, n_clusters):

    # Cargar los datos
    matrix = load_matrix(matrix_file)
    barcodes = load_barcodes(barcodes_file)
    genes = load_genes(genes_file)
    
    # Explorar los datos
    explore_data(matrix, barcodes, genes)
    #filtrar datos
    filtered_matrix, filtered_barcodes, filtered_genes = filter_data(matrix, barcodes, genes)
    #filtered_matrix, filtered_barcodes, filtered_genes = filter_interactively(matrix, barcodes, genes)

    # Normalizar datos
    # normalized_matrix = normalize_data(filtered_matrix)
    print("\nProcesando y normalizando datos...")
    normalized_matrix, hvg_matrix, hvg_genes, gene_stats = process_normalized_data(
        filtered_matrix,
        #matrix, 
        filtered_genes
        #genes,
        #n_top_genes=500
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

    # print("\nRealizando clustering con KMeans...")
    # cluster_labels = perform_clustering(pca_result, n_clusters)
    # Realizar clustering usando Pearson
    print("\nRealizando clustering con KMeans basado en Pearson...")
    cluster_labels, centers = kmeans_with_pearson(pca_result, n_clusters=n_clusters)

    print("Clustering realizado. Visualizando clusters...")
     # Visualización de Clustering
    plot_clusters(pca_result, cluster_labels)

    community_labels = label_propagation(pca_result, n_neighbors=10)
    save_clustering_results(pca_result, cluster_labels)
    plot_clusters_with_labels(pca_result, community_labels)
    #plot_clusters(pca_result, community_labels)
    calculate_silhouette_score(pca_result, cluster_labels)






    print("Guardando resultados clusters...")
    save_clustering_results(pca_result, cluster_labels)
    #calculate_silhouette_score(pca_result, cluster_labels)

    







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algoritmo de clustering de single cell")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")
    parser.add_argument("--n_clusters", type=int, default=10, help="Número de clusters a generar (k)")
    

    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file, args.n_clusters)