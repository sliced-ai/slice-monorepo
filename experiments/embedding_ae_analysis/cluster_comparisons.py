import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.cluster import KMeans
from cuml.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_embeddings(file_path, encoded=False):
    with h5py.File(file_path, 'r') as f:
        if encoded:
            embeddings = f['encoded_embeddings'][:]
        else:
            embeddings = f['embeddings'][:]
        names = f['names'][:]
        turn_numbers = f['turn_numbers'][:]
        file_paths = f['file_paths'][:]
        model_names = f['model_names'][:]
    
    metadata = {
        'names': [name.decode('utf8') for name in names],
        'turn_numbers': turn_numbers,
        'file_paths': [file_path.decode('utf8') for file_path in file_paths],
        'model_names': [model_name.decode('utf8') for model_name in model_names]
    }
    return embeddings, metadata

def visualize_tsne(embeddings, clusters, title, file_name):
    tsne = TSNE(n_components=2)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'tsne_x': tsne_embeddings[:, 0],
        'tsne_y': tsne_embeddings[:, 1],
        'cluster': clusters
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster', data=df, palette='tab20', s=5)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()

def calculate_clustering_metrics(embeddings, clusters):
    silhouette_avg = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    return silhouette_avg, davies_bouldin

def main(custom_name):
    # Create experiment directory
    output_folder = os.path.join('experiments', custom_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load embeddings
    raw_embeddings_path = 'utterance_embeddings_ds1.h5'
    encoded_embeddings_path = '/workspace/slice-monorepo/cl_cr3/high_accuracy_v1_visuals. - 2024-06-22/encoded_embeddings.h5'

    raw_embeddings, raw_metadata = load_embeddings(raw_embeddings_path)
    encoded_embeddings, encoded_metadata = load_embeddings(encoded_embeddings_path, encoded=True)

    # Ensure compatibility (trimming to the same size if needed)
    min_length = min(len(raw_embeddings), len(encoded_embeddings))
    raw_embeddings = raw_embeddings[:min_length]
    encoded_embeddings = encoded_embeddings[:min_length]

    # Standardize embeddings for clustering
    scaler = StandardScaler()
    raw_embeddings = scaler.fit_transform(raw_embeddings)
    encoded_embeddings = scaler.fit_transform(encoded_embeddings)

    # Perform KMeans clustering
    n_clusters = 10
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_raw.fit(raw_embeddings)
    raw_clusters = kmeans_raw.labels_

    kmeans_encoded = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_encoded.fit(encoded_embeddings)
    encoded_clusters = kmeans_encoded.labels_

    # Calculate clustering metrics
    raw_silhouette, raw_davies_bouldin = calculate_clustering_metrics(raw_embeddings, raw_clusters)
    encoded_silhouette, encoded_davies_bouldin = calculate_clustering_metrics(encoded_embeddings, encoded_clusters)

    metrics = {
        'raw_embeddings': {
            'silhouette_score': raw_silhouette,
            'davies_bouldin_index': raw_davies_bouldin
        },
        'encoded_embeddings': {
            'silhouette_score': encoded_silhouette,
            'davies_bouldin_index': encoded_davies_bouldin
        }
    }

    with open(os.path.join(output_folder, 'clustering_comparison_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Visualize using t-SNE
    visualize_tsne(raw_embeddings, raw_clusters, 'Raw Embeddings t-SNE Clustering', os.path.join(output_folder, 'raw_embeddings_tsne.png'))
    visualize_tsne(encoded_embeddings, encoded_clusters, 'Encoded Embeddings t-SNE Clustering', os.path.join(output_folder, 'encoded_embeddings_tsne.png'))

    # Print results
    print(f"Raw Embeddings: Silhouette Score = {raw_silhouette}, Davies-Bouldin Index = {raw_davies_bouldin}")
    print(f"Encoded Embeddings: Silhouette Score = {encoded_silhouette}, Davies-Bouldin Index = {encoded_davies_bouldin}")

if __name__ == "__main__":
    custom_name = 'high_accuracy_v1_visuals'  # Update this with your custom name
    main(custom_name)
