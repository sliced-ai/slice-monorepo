import os
import json
import logging
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
from cuml.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from datetime import datetime
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Loading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertAutoencoder('bert-base-uncased', embedding_dim=1536).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded successfully.")
    return model

def load_embeddings(file_path):
    logging.info(f"Loading embeddings from {file_path}...")
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
    logging.info("Embeddings loaded successfully.")
    return embeddings, metadata

def save_encoded_embeddings_to_hdf5(encoded_embeddings, metadata, filename):
    logging.info(f"Saving encoded embeddings to {filename}...")
    with h5py.File(filename, 'w') as f:
        f.create_dataset('encoded_embeddings', data=encoded_embeddings)
        f.create_dataset('names', data=np.string_(metadata['names']))
        f.create_dataset('turn_numbers', data=metadata['turn_numbers'])
        f.create_dataset('file_paths', data=np.string_(metadata['file_paths']))
        f.create_dataset('model_names', data=np.string_(metadata['model_names']))
    logging.info("Encoded embeddings saved successfully.")

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
    if len(set(clusters)) > 1:
        silhouette_avg = silhouette_score(embeddings, clusters)
        ari = adjusted_rand_score(clusters, clusters)  # Should be between true labels and predicted labels
        nmi = normalized_mutual_info_score(clusters, clusters)  # Should be between true labels and predicted labels
    else:
        silhouette_avg, ari, nmi = -1, -1, -1
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
    logging.info("Starting the process...")
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
    logging.info("Encoding embeddings using the autoencoder...")
    all_embeddings_encoded = []
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for inputs in dataloader:
        inputs = inputs[0].to(device)
        encoded_embeddings = model.encode(inputs)
        all_embeddings_encoded.append(encoded_embeddings.cpu().numpy())
    
    all_embeddings_encoded = np.vstack(all_embeddings_encoded)
    logging.info("Embeddings encoded successfully.")

    # Clear the GPU memory
    del model
    torch.cuda.empty_cache()
    logging.info("Cleared model and GPU cache.")

    # Clear RAM memory of the original embeddings to free up space
    del embeddings
    logging.info("Cleared original embeddings from RAM.")

    # Ensure that the number of labels matches the number of embeddings
    min_length = min(len(metadata['names']), len(all_embeddings_encoded))
    metadata['names'] = metadata['names'][:min_length]
    metadata['turn_numbers'] = metadata['turn_numbers'][:min_length]
    all_embeddings_encoded = all_embeddings_encoded[:min_length]

    # Summarize encoded embeddings
    summarize_data(all_embeddings_encoded, os.path.join(output_folder, 'encoded_embeddings'))

    # Save encoded embeddings to an HDF5 file
    save_encoded_embeddings_to_hdf5(all_embeddings_encoded, metadata, os.path.join(output_folder, 'encoded_embeddings.h5'))

    # Dimensionality reduction with GPU UMAP for encoded embeddings
    logging.info("Reducing dimensions using UMAP with GPU for encoded embeddings...")
    reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings_encoded = reducer.fit_transform(all_embeddings_encoded)
    logging.info("UMAP reduction for encoded embeddings completed.")

    # Perform DBSCAN clustering with GPU for encoded embeddings
    logging.info("Performing DBSCAN clustering with GPU for encoded embeddings...")
    dbscan_encoded = DBSCAN(eps=0.5, min_samples=5)
    clusters_encoded = dbscan_encoded.fit_predict(umap_embeddings_encoded)
    logging.info("DBSCAN clustering for encoded embeddings completed.")

    # Convert to DataFrame for easier handling
    df_encoded = pd.DataFrame({
        'name': metadata['names'],
        'turn_number': metadata['turn_numbers'],
        'umap_x': umap_embeddings_encoded[:, 0],
        'umap_y': umap_embeddings_encoded[:, 1],
        'cluster': clusters_encoded
    })

    # Visualize the UMAP embeddings colored by names for encoded embeddings
    visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'name', 'Encoded UMAP Embeddings Colored by Names', os.path.join(output_folder, 'encoded_umap_colored_by_names.png'))

    # Visualize the UMAP embeddings with clusters for encoded embeddings
    visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'cluster', 'UMAP projection with DBSCAN clusters (Encoded)', os.path.join(output_folder, 'umap_dbscan_clusters_encoded.png'))

    # Calculate and print cluster metrics using parallel processing
    embeddings_list = [umap_embeddings_encoded]
    clusters_list = [clusters_encoded]
    
    results = calculate_all_metrics(embeddings_list, clusters_list, n_jobs=4)

    metrics = {
        'umap_encoded': {
            'silhouette_score': float(results[0][0]),
            'ari': float(results[0][1]),
            'nmi': float(results[0][2])
        }
    }

    with open(os.path.join(output_folder, 'cluster_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Encoded UMAP Embeddings: Silhouette Score = {results[0][0]}, ARI = {results[0][1]}, NMI = {results[0][2]}")

if __name__ == "__main__":
    config = {
        'experiment_name': 'high_accuarcy_dbscan',
        'embeddings_path': 'utterance_embeddings_ds1.h5',
        'model_path': 'trained_autoencoder.pth',
        'batch_size': 32768
    }
    main(config)
