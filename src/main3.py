import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
import psutil  # Para monitoreo de memoria

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

# --- 2. Exploración y resumen de datos
def explore_data(matrix, barcodes, genes):
    print("Dimensiones de la matriz de expresión (genes x células):", matrix.shape)
    print("Número de barcodes (células):", barcodes.shape[0])
    print("Número de genes:", genes.shape[0])

# --- 3. Normalización
def normalize_data(matrix, scale_factor=10000):
    cell_sums = matrix.sum(axis=0).A1
    cell_sums = np.where(cell_sums == 0, 1, cell_sums)
    scaling_factors = scale_factor / cell_sums
    normalized_matrix = matrix.copy()
    normalized_matrix = normalized_matrix.multiply(scaling_factors[np.newaxis, :])
    normalized_matrix.data = np.log1p(normalized_matrix.data)
    return normalized_matrix.tocsc()

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
    return labels

def adjust_clustering_with_modularity(data, labels):
    similarity_matrix = 1 - pairwise_distances(data, metric="cosine")
    graph = nx.from_numpy_array(similarity_matrix)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    modularity = nx.algorithms.community.quality.modularity(graph, communities)
    print(f"Modularidad final ajustada: {modularity:.4f}")
    return labels, modularity

# --- 6. Métricas de memoria y Silhouette Score
def log_metrics(data, labels, modularity):
    process = psutil.Process()
    memory_used = process.memory_info().rss / (1024 ** 2)  # Convertir a MB
    silhouette = silhouette_score(data, labels, metric="cosine")
    print(f"Uso de memoria: {memory_used:.2f} MB")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Modularidad final: {modularity:.4f}")

# --- 7. Guardar resultados
def save_results(labels, filename="clustering_results.csv"):
    df = pd.DataFrame({"Cluster": labels})
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}.")

# --- Función principal
def main(matrix_file, barcodes_file, genes_file):
    matrix = load_matrix(matrix_file)
    barcodes = load_barcodes(barcodes_file)
    genes = load_genes(genes_file)

    explore_data(matrix, barcodes, genes)

    normalized_matrix = normalize_data(matrix)
    highly_variable = normalized_matrix.mean(axis=1).A1 > np.percentile(normalized_matrix.mean(axis=1).A1, 85)
    hvg_matrix = normalized_matrix[highly_variable, :]

    pca_result = apply_pca(hvg_matrix, n_components=10)
    cosine_labels = perform_cosine_clustering(pca_result, n_clusters=10)
    adjusted_labels, modularity = adjust_clustering_with_modularity(pca_result, cosine_labels)
    
    # Logear métricas
    log_metrics(pca_result, adjusted_labels, modularity)

    # Guardar resultados
    save_results(adjusted_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering con análisis de modularidad")
    parser.add_argument("--matrix_file", required=True, help="Ruta del archivo matrix.mtx")
    parser.add_argument("--barcodes_file", required=True, help="Ruta del archivo barcodes.tsv")
    parser.add_argument("--genes_file", required=True, help="Ruta del archivo genes.tsv")
    args = parser.parse_args()
    main(args.matrix_file, args.barcodes_file, args.genes_file)
