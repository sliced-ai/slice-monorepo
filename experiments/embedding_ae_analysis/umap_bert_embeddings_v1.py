import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
from cuml.manifold import UMAP as cumlUMAP
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
import pandas as pd
import seaborn as sns
from cuml.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from datetime import datetime
from joblib import Parallel, delayed

# Define the Autoencoder model
class BertAutoencoder(nn.Module):
    def __init__(self, bert_model_name, embedding_dim=1536, hidden_dim=768, lstm_units=256, sequence_length=4):
        super(BertAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_adapter = nn.Linear(embedding_dim, hidden_dim * sequence_length)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.decoder = nn.LSTM(hidden_dim, lstm_units, batch_first=True)
        self.output_layer = nn.Linear(lstm_units * sequence_length, embedding_dim)

    def forward(self, x):
        x = self.embedding_adapter(x)
        x = x.view(x.size(0), self.sequence_length, -1)  # Reshape to (batch_size, sequence_length, hidden_dim)
        encoder_outputs = self.bert(inputs_embeds=x).last_hidden_state
        decoder_outputs, _ = self.decoder(encoder_outputs)
        decoder_outputs = decoder_outputs.contiguous().view(decoder_outputs.size(0), -1)  # Flatten to match input dimensions
        output = self.output_layer(decoder_outputs)
        return output

    def encode(self, x):
        x = self.embedding_adapter(x)
        x = x.view(x.size(0), self.sequence_length, -1)  # Reshape to (batch_size, sequence_length, hidden_dim)
        with torch.no_grad():
            encoder_outputs = self.bert(inputs_embeds=x).last_hidden_state
        return encoder_outputs[:, 0, :]  # Return the [CLS] token embedding

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertAutoencoder('bert-base-uncased', embedding_dim=1536).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
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

def save_encoded_embeddings_to_hdf5(encoded_embeddings, metadata, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('encoded_embeddings', data=encoded_embeddings)
        f.create_dataset('names', data=np.string_(metadata['names']))
        f.create_dataset('turn_numbers', data=metadata['turn_numbers'])
        f.create_dataset('file_paths', data=np.string_(metadata['file_paths']))
        f.create_dataset('model_names', data=np.string_(metadata['model_names']))

def visualize_embeddings(df, x_col, y_col, hue_col, title, file_name):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='tab20', s=5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(file_name)
    plt.show()

def calculate_cluster_metrics(embeddings, clusters):
    silhouette_avg = silhouette_score(embeddings, clusters)
    ari = adjusted_rand_score(clusters, clusters)  # Should be between true labels and predicted labels
    nmi = normalized_mutual_info_score(clusters, clusters)  # Should be between true labels and predicted labels
    return silhouette_avg, ari, nmi

def calculate_all_metrics(embeddings_list, clusters_list, n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_cluster_metrics)(embeddings, clusters) 
                                      for embeddings, clusters in zip(embeddings_list, clusters_list))
    return results

def summarize_data(data, name):
    summary = {
        'shape': data.shape,
        'mean': np.mean(data, axis=0).tolist(),
        'std': np.std(data, axis=0).tolist(),
        'min': np.min(data, axis=0).tolist(),
        'max': np.max(data, axis=0).tolist()
    }
    with open(f'{name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

def main(config):
    experiment_name = config['experiment_name']
    embeddings_path = config['embeddings_path']
    model_path = config['model_path']
    batch_size = config['batch_size']
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_folder = f"{experiment_name} - {date_str}"
    os.makedirs(output_folder, exist_ok=True)

    embeddings, metadata = load_embeddings(embeddings_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    model.eval()
    
    # Encode embeddings using the autoencoder
    all_embeddings_encoded = []
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for inputs in dataloader:
        inputs = inputs[0].to(device)
        encoded_embeddings = model.encode(inputs)
        all_embeddings_encoded.append(encoded_embeddings.cpu().numpy())
    
    all_embeddings_encoded = np.vstack(all_embeddings_encoded)

    # Ensure that the number of labels matches the number of embeddings
    min_length = min(len(metadata['names']), len(embeddings), len(all_embeddings_encoded))
    metadata['names'] = metadata['names'][:min_length]
    metadata['turn_numbers'] = metadata['turn_numbers'][:min_length]
    embeddings = embeddings[:min_length]
    all_embeddings_encoded = all_embeddings_encoded[:min_length]

    # Summarize raw and encoded embeddings
    summarize_data(embeddings, os.path.join(output_folder, 'raw_embeddings'))
    summarize_data(all_embeddings_encoded, os.path.join(output_folder, 'encoded_embeddings'))

    # Save encoded embeddings to an HDF5 file
    save_encoded_embeddings_to_hdf5(all_embeddings_encoded, metadata, os.path.join(output_folder, 'encoded_embeddings.h5'))

    # Dimensionality reduction with GPU UMAP for raw embeddings
    print("Reducing dimensions using UMAP with GPU for raw embeddings...")
    reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings_raw = reducer.fit_transform(embeddings)

    # Perform KMeans clustering with GPU for raw embeddings
    print("Performing KMeans clustering with GPU for raw embeddings...")
    n_clusters = 10
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_raw.fit(umap_embeddings_raw)
    clusters_raw = kmeans_raw.labels_

    # Dimensionality reduction with GPU UMAP for encoded embeddings
    print("Reducing dimensions using UMAP with GPU for encoded embeddings...")
    umap_embeddings_encoded = reducer.fit_transform(all_embeddings_encoded)

    # Perform KMeans clustering with GPU for encoded embeddings
    print("Performing KMeans clustering with GPU for encoded embeddings...")
    kmeans_encoded = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_encoded.fit(umap_embeddings_encoded)
    clusters_encoded = kmeans_encoded.labels_

    # PCA for raw embeddings
    print("Reducing dimensions using PCA for raw embeddings...")
    pca_raw = PCA(n_components=2)
    pca_embeddings_raw = pca_raw.fit_transform(embeddings)

    # Perform KMeans clustering for PCA raw embeddings
    print("Performing KMeans clustering for PCA raw embeddings...")
    kmeans_pca_raw = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_pca_raw.fit(pca_embeddings_raw)
    clusters_pca_raw = kmeans_pca_raw.labels_

    # PCA for encoded embeddings
    print("Reducing dimensions using PCA for encoded embeddings...")
    pca_encoded = PCA(n_components=2)
    pca_embeddings_encoded = pca_encoded.fit_transform(all_embeddings_encoded)

    # Perform KMeans clustering for PCA encoded embeddings
    print("Performing KMeans clustering for PCA encoded embeddings...")
    kmeans_pca_encoded = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_pca_encoded.fit(pca_embeddings_encoded)
    clusters_pca_encoded = kmeans_pca_encoded.labels_

    # Convert to DataFrame for easier handling
    df_raw = pd.DataFrame({
        'name': metadata['names'],
        'turn_number': metadata['turn_numbers'],
        'umap_x': umap_embeddings_raw[:, 0],
        'umap_y': umap_embeddings_raw[:, 1],
        'cluster': clusters_raw
    })

    df_encoded = pd.DataFrame({
        'name': metadata['names'],
        'turn_number': metadata['turn_numbers'],
        'umap_x': umap_embeddings_encoded[:, 0],
        'umap_y': umap_embeddings_encoded[:, 1],
        'cluster': clusters_encoded
    })

    df_pca_raw = pd.DataFrame({
        'name': metadata['names'],
        'turn_number': metadata['turn_numbers'],
        'pca_x': pca_embeddings_raw[:, 0],
        'pca_y': pca_embeddings_raw[:, 1],
        'cluster': clusters_pca_raw
    })

    df_pca_encoded = pd.DataFrame({
        'name': metadata['names'],
        'turn_number': metadata['turn_numbers'],
        'pca_x': pca_embeddings_encoded[:, 0],
        'pca_y': pca_embeddings_encoded[:, 1],
        'cluster': clusters_pca_encoded
    })

    # Visualize the UMAP embeddings colored by names for raw and encoded embeddings
    visualize_embeddings(df_raw, 'umap_x', 'umap_y', 'name', 'Raw UMAP Embeddings Colored by Names', os.path.join(output_folder, 'raw_umap_colored_by_names.png'))
    visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'name', 'Encoded UMAP Embeddings Colored by Names', os.path.join(output_folder, 'encoded_umap_colored_by_names.png'))

    # Visualize the UMAP embeddings with clusters for raw and encoded embeddings
    visualize_embeddings(df_raw, 'umap_x', 'umap_y', 'cluster', 'UMAP projection with KMeans clusters (Raw)', os.path.join(output_folder, 'umap_kmeans_clusters_raw.png'))
    visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'cluster', 'UMAP projection with KMeans clusters (Encoded)', os.path.join(output_folder, 'umap_kmeans_clusters_encoded.png'))

    # Visualize the PCA embeddings colored by names for raw and encoded embeddings
    visualize_embeddings(df_pca_raw, 'pca_x', 'pca_y', 'name', 'Raw PCA Embeddings Colored by Names', os.path.join(output_folder, 'raw_pca_colored_by_names.png'))
    visualize_embeddings(df_pca_encoded, 'pca_x', 'pca_y', 'name', 'Encoded PCA Embeddings Colored by Names', os.path.join(output_folder, 'encoded_pca_colored_by_names.png'))

    # Visualize the PCA embeddings with clusters for raw and encoded embeddings
    visualize_embeddings(df_pca_raw, 'pca_x', 'pca_y', 'cluster', 'PCA projection with KMeans clusters (Raw)', os.path.join(output_folder, 'pca_kmeans_clusters_raw.png'))
    visualize_embeddings(df_pca_encoded, 'pca_x', 'pca_y', 'cluster', 'PCA projection with KMeans clusters (Encoded)', os.path.join(output_folder, 'pca_kmeans_clusters_encoded.png'))

    # Calculate and print cluster metrics using parallel processing
    embeddings_list = [umap_embeddings_raw, umap_embeddings_encoded, pca_embeddings_raw, pca_embeddings_encoded]
    clusters_list = [clusters_raw, clusters_encoded, clusters_pca_raw, clusters_pca_encoded]
    
    results = calculate_all_metrics(embeddings_list, clusters_list, n_jobs=4)

    metrics = {
        'umap_raw': {
            'silhouette_score': float(results[0][0]),
            'ari': float(results[0][1]),
            'nmi': float(results[0][2])
        },
        'umap_encoded': {
            'silhouette_score': float(results[1][0]),
            'ari': float(results[1][1]),
            'nmi': float(results[1][2])
        },
        'pca_raw': {
            'silhouette_score': float(results[2][0]),
            'ari': float(results[2][1]),
            'nmi': float(results[2][2])
        },
        'pca_encoded': {
            'silhouette_score': float(results[3][0]),
            'ari': float(results[3][1]),
            'nmi': float(results[3][2])
        }
    }

    with open(os.path.join(output_folder, 'cluster_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Raw UMAP Embeddings: Silhouette Score = {results[0][0]}, ARI = {results[0][1]}, NMI = {results[0][2]}")
    print(f"Encoded UMAP Embeddings: Silhouette Score = {results[1][0]}, ARI = {results[1][1]}, NMI = {results[1][2]}")
    print(f"Raw PCA Embeddings: Silhouette Score = {results[2][0]}, ARI = {results[2][1]}, NMI = {results[2][2]}")
    print(f"Encoded PCA Embeddings: Silhouette Score = {results[3][0]}, ARI = {results[3][1]}, NMI = {results[3][2]}")

if __name__ == "__main__":
    config = {
        'experiment_name': 'high_accuracy_v1_visuals.',  # Update this with your experiment name
        'embeddings_path': 'utterance_embeddings_ds1.h5',
        'model_path': 'trained_autoencoder.pth',
        'batch_size': 32768
    }
    main(config)
