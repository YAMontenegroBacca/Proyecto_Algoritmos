import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Leer datos
def load_matrix(matrix_file):
    matrix = mmread(matrix_file).tocsc()
    return matrix

def load_barcodes(barcodes_file):
    barcodes = pd.read_csv(barcodes_file, header=None, sep='\t')
    barcodes.columns = ['barcode']
    return barcodes

def load_genes(genes_file):
    genes = pd.read_csv(genes_file, header=None, sep='\t')
    genes.columns = ['gene_id', 'gene_name']
    return genes

# --- 2. Explorar datos
def explore_data(matrix, barcodes, genes):
    print("Dimensiones de la matriz de expresión (genes x células):", matrix.shape)
    print("Número de barcodes (células):", barcodes.shape[0])
    print("Número de genes:", genes.shape[0])

def display_data_summary(matrix, genes):
    gene_expression_sum = matrix.sum(axis=1).A1
    stats = {
        'max_expression': np.max(gene_expression_sum),
        'min_expression': np.min(gene_expression_sum),
        'mean_expression': np.mean(gene_expression_sum),
    }
    print("\nResumen de datos:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")

    highly_variable_genes = calculate_highly_variable_genes(matrix, percentile=85)[1]
    print("\nGenes altamente variables (percentil 85):")
    print(highly_variable_genes[highly_variable_genes['highly_variable']].head())


# --- 3. Filtrar y Normalizar
def normalize_data(matrix, scale_factor=10000):
    cell_sums = matrix.sum(axis=0).A1
    cell_sums = np.where(cell_sums == 0, 1, cell_sums)
    scaling_factors = scale_factor / cell_sums
    normalized_matrix = matrix.copy()
    normalized_matrix = normalized_matrix.multiply(scaling_factors[np.newaxis, :])
    normalized_matrix = normalized_matrix.tocoo()
    normalized_matrix.data = np.log1p(normalized_matrix.data)
    return normalized_matrix.tocsc()

# def calculate_highly_variable_genes(normalized_matrix, n_top_genes=100):
#     means = np.array(normalized_matrix.mean(axis=1)).flatten()
#     vars = np.array(normalized_matrix.power(2).mean(axis=1)).flatten() - means**2
#     stats = pd.DataFrame({
#         'mean': means,
#         'var': vars,
#         'cv2': vars / (means**2 + 1e-6)
#     })
#     highly_variable = np.zeros(normalized_matrix.shape[0], dtype=bool)
#     highly_variable[np.argsort(stats['cv2'])[-n_top_genes:]] = True
#     stats['highly_variable'] = highly_variable
#     return highly_variable, stats

def calculate_highly_variable_genes(normalized_matrix, percentile=85):
    means = np.array(normalized_matrix.mean(axis=1)).flatten()
    vars = np.array(normalized_matrix.power(2).mean(axis=1)).flatten() - means**2
    stats = pd.DataFrame({
        'mean': means,
        'var': vars,
        'cv2': vars / (means**2 + 1e-6)
    })
    threshold = np.percentile(stats['cv2'], percentile)
    highly_variable = stats['cv2'] >= threshold
    stats['highly_variable'] = highly_variable
    num_highly_variable = np.sum(highly_variable)
    print(f"\nNúmero de genes seleccionados como altamente variables: {num_highly_variable}")
    return highly_variable, stats

# --- 4. PCA
def apply_pca(hvg_matrix, n_components=100):
    dense_matrix = hvg_matrix.toarray()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(dense_matrix.T)
    return pca_result

# --- 5. Clustering
def perform_cosine_clustering(data, n_clusters=10):
    distance_matrix = 1 - pairwise_distances(data, metric="cosine")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(distance_matrix)
    print(f"Clustering con distancia de coseno completado. Número de clusters: {n_clusters}")
    return labels

# def adjust_clustering_with_modularity(data, labels):
#     similarity_matrix = 1 - pairwise_distances(data, metric="euclidean")
#     #graph = nx.from_numpy_matrix(similarity_matrix)
#     graph = nx.from_numpy_array(similarity_matrix)
#     communities = {i: set(np.where(labels == i)[0]) for i in set(labels)}
#     modularity_initial = nx.algorithms.community.quality.modularity(graph, communities.values())
#     print(f"Modularidad inicial: {modularity_initial:.4f}")
#     partitions = nx.algorithms.community.greedy_modularity_communities(graph)
#     adjusted_labels = np.zeros_like(labels)
#     for i, community in enumerate(partitions):
#         for node in community:
#             adjusted_labels[node] = i
#     modularity_final = nx.algorithms.community.quality.modularity(graph, partitions)
#     print(f"Modularidad final ajustada: {modularity_final:.4f}")
#     return adjusted_labels

def adjust_clustering_with_modularity(data, labels, n_clusters=10):
    """
    Ajusta la asignación de clusters usando modularidad y reduce a n_clusters exactos
    mediante KMeans con distancia de coseno.
    
    Args:
        data: matriz de datos (filas: muestras, columnas: características).
        labels: etiquetas de cluster originales.
        n_clusters: número fijo de clusters deseados.

    Returns:
        final_labels: etiquetas ajustadas a exactamente n_clusters.
    """
    similarity_matrix = 1 - pairwise_distances(data, metric="cosine")
    graph = nx.from_numpy_array(similarity_matrix)
    
    # Encontrar comunidades iniciales con modularidad
    partitions = list(nx.algorithms.community.greedy_modularity_communities(graph))
    print(f"Comunidades detectadas por modularidad: {len(partitions)}")
    
    # Crear una representación de nodos -> etiquetas de comunidad
    node_to_community = np.zeros(len(labels), dtype=int)
    for i, community in enumerate(partitions):
        for node in community:
            node_to_community[node] = i
    
    # Reagrupar comunidades detectadas con KMeans usando distancia de coseno
    if len(partitions) > n_clusters:
        # Calcular centroides de las comunidades
        community_centers = []
        for i in range(len(partitions)):
            nodes_in_community = np.where(node_to_community == i)[0]
            community_data = data[nodes_in_community]
            community_centers.append(community_data.mean(axis=0))
        community_centers = np.array(community_centers)
        
        # Aplicar KMeans con distancia de coseno sobre los centroides
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        reduced_labels = kmeans.fit_predict(community_centers)

        # Reasignar nodos a clusters finales
        final_labels = np.zeros_like(labels)
        for i, community in enumerate(partitions):
            for node in community:
                final_labels[node] = reduced_labels[i]
        print(f"Clusters finales reducidos a {n_clusters} utilizando KMeans con distancia de coseno.")
        return final_labels
    else:
        print(f"Modularidad generó menos de {n_clusters} comunidades, manteniendo resultados originales.")
        return node_to_community


# --- 6. Guardar resultados
def save_results(labels, filename="clustering_results_cosine2.csv"):
    df = pd.DataFrame({"Cluster": labels})
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}.")

# --- 7. Graficar resultados
# def plot_clusters(pca_result, labels, filename="clusters_plot_cosine2.png"):
#     plt.figure(figsize=(10, 8))
#     unique_labels = np.unique(labels)
#     for label in unique_labels:
#         cluster_points = pca_result[labels == label]
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.6)
#     plt.title("Clustering en el espacio PCA")
#     plt.xlabel("Componente principal 1")
#     plt.ylabel("Componente principal 2")
#     plt.legend()
#     plt.savefig(filename)
#     print(f"Gráfica de clusters guardada en {filename}.")
def plot_clusters(pca_result, labels, filename="clusters_plot.png", title="Clustering en el espacio PCA"):
    """
    Genera un gráfico de los clusters en el espacio PCA y lo guarda en un archivo.

    Args:
        pca_result: Resultado de PCA (matriz de 2 dimensiones).
        labels: Etiquetas de clustering para cada punto.
        filename: Nombre del archivo donde se guarda la gráfica.
        title: Título de la gráfica.
    """
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = pca_result[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.6)
    plt.title(title)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.savefig(filename)
    print(f"Gráfica guardada en {filename}.")



# --- Función principal
def main(matrix_file, barcodes_file, genes_file):
    matrix = load_matrix(matrix_file)
    barcodes = load_barcodes(barcodes_file)
    genes = load_genes(genes_file)

    explore_data(matrix, barcodes, genes)
    display_data_summary(matrix, genes)
    
    normalized_matrix = normalize_data(matrix)
    highly_variable, gene_stats = calculate_highly_variable_genes(normalized_matrix)
    hvg_matrix = normalized_matrix[highly_variable, :]

    pca_result = apply_pca(hvg_matrix, n_components=10)
    cosine_labels = perform_cosine_clustering(pca_result, n_clusters=10)
      # Gráfica inicial (antes de modularidad)
    plot_clusters(pca_result, cosine_labels, filename="clusters_initial.png", 
                  title="Clustering inicial (antes de modularidad)")
    adjusted_labels = adjust_clustering_with_modularity(pca_result, cosine_labels)
    
    # Gráfica ajustada (después de modularidad)
    plot_clusters(pca_result, adjusted_labels, filename="clusters_adjusted.png", 
                  title="Clustering ajustado (después de modularidad)")
    save_results(adjusted_labels)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering con análisis de modularidad")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")
    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file)
