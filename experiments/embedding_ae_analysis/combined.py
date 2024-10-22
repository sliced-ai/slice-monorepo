import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from transformers import BertModel
import matplotlib.pyplot as plt
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import seaborn as sns
from cuml.manifold import UMAP as cumlUMAP
from cuml.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

class BertAutoencoder(nn.Module):
    def __init__(self, device, bert_model_name='bert-base-uncased', embedding_dim=1536, hidden_dim=768, lstm_units=256, sequence_length=4):
        super(BertAutoencoder, self).__init__()
        self.device = device
        self.sequence_length = sequence_length
        self.embedding_adapter = nn.Linear(embedding_dim, hidden_dim * sequence_length)
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.decoder = nn.LSTM(hidden_dim, lstm_units, batch_first=True).to(device)
        self.output_layer = nn.Linear(lstm_units * sequence_length, embedding_dim).to(device)

    def forward(self, x):
        x = self.embedding_adapter(x).to(self.device)
        x = x.view(x.size(0), self.sequence_length, -1)  # Reshape to (batch_size, sequence_length, hidden_dim)
        encoder_outputs = self.bert(inputs_embeds=x).last_hidden_state
        decoder_outputs, _ = self.decoder(encoder_outputs)
        decoder_outputs = decoder_outputs.contiguous().view(decoder_outputs.size(0), -1)  # Flatten to match input dimensions
        output = self.output_layer(decoder_outputs)
        return output

    def encode(self, x):
        x = self.embedding_adapter(x).to(self.device)
        x = x.view(x.size(0), self.sequence_length, -1)  # Reshape to (batch_size, sequence_length, hidden_dim)
        with torch.no_grad():
            encoder_outputs = self.bert(inputs_embeds=x).last_hidden_state
        return encoder_outputs[:, 0, :]  # Return the [CLS] token embedding

class Trainer:
    def __init__(self, model, initial_size=4096, increment_ratio=0.005, max_epochs=50000, lr=0.0001):
        self.model = model
        self.initial_size = initial_size
        self.increment_ratio = increment_ratio
        self.max_epochs = max_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = model.device

    def load_embeddings(self, file_path):
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

    def calculate_cosine_similarity(self, outputs, inputs):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(outputs, inputs)
        return similarities.mean().item() * 100  # Convert to percentage

    class Prefetcher:
        def __init__(self, loader):
            self.loader = iter(loader)
            self.stream = torch.cuda.Stream()
            self.next_input = None
            self.preload()

        def preload(self):
            try:
                self.next_input = next(self.loader)
            except StopIteration:
                self.next_input = None
                return
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input[0].cuda(non_blocking=True)

        def next(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            self.preload()
            return input

    def prepare_dataloader(self, train_dataset, current_size, batch_size, num_workers=4):
        current_train_indices = torch.randperm(len(train_dataset))[:int(current_size)]
        current_train_dataset = Subset(train_dataset, current_train_indices)
        train_loader = DataLoader(current_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        return train_loader

    def progressive_training(self, train_dataset):
        total_size = len(train_dataset)
        test_size = 0.1
        metrics = []

        # Separate the test data and ensure it's shuffled
        train_size = int((1 - test_size) * total_size)
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=self.initial_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

        current_size = self.initial_size

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_train_loader = executor.submit(self.prepare_dataloader, train_dataset, current_size, self.initial_size, num_workers=8)
            
            for epoch in range(self.max_epochs):
                if current_size > train_size:
                    break

                train_loader = future_train_loader.result()
                prefetcher = self.Prefetcher(train_loader)
                future_train_loader = executor.submit(self.prepare_dataloader, train_dataset, current_size * (1 + self.increment_ratio), self.initial_size, num_workers=8)

                self.model.train()
                train_loss, train_similarity = 0, 0

                input = prefetcher.next()
                while input is not None:
                    self.optimizer.zero_grad()
                    outputs = self.model(input)
                    loss = self.criterion(outputs, input)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    train_similarity += self.calculate_cosine_similarity(outputs, input)
                    input = prefetcher.next()

                train_loss /= len(train_loader)
                train_similarity /= len(train_loader)

                self.model.eval()
                val_loss, val_similarity = 0, 0
                with torch.no_grad():
                    for inputs in test_loader:
                        inputs = inputs[0].to(self.device, non_blocking=True)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        val_loss += loss.item()
                        val_similarity += self.calculate_cosine_similarity(outputs, inputs)

                val_loss /= len(test_loader)
                val_similarity /= len(test_loader)

                metrics.append((current_size, train_loss, val_loss, train_similarity, val_similarity))
                current_size += current_size * self.increment_ratio

                print(f'Data size: {int(current_size)} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Train Similarity: {train_similarity:.2f}% | Val Similarity: {val_similarity:.2f}%')
        
        return metrics

    def save_metrics(self, metrics, save_path):
        sizes, train_losses, val_losses, train_similarities, val_similarities = zip(*metrics)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(sizes, train_losses, label='Training Loss')
        plt.plot(sizes, val_losses, label='Validation Loss')
        plt.xlabel('Data Size')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs Data Size')

        plt.subplot(2, 1, 2)
        plt.plot(sizes, train_similarities, label='Training Similarity')
        plt.plot(sizes, val_similarities, label='Validation Similarity')
        plt.xlabel('Data Size')
        plt.ylabel('Similarity (%)')
        plt.legend()
        plt.title('Similarity vs Data Size')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'metrics.png'))
        plt.close()

        # Save raw data
        metrics_data = {
            'sizes': sizes,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_similarities': train_similarities,
            'val_similarities': val_similarities
        }

        with open(os.path.join(save_path, 'metrics_data.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)

class Analysis:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def summarize_data(self, data, name):
        per_vector_means = np.mean(data, axis=1)
        per_vector_stds = np.std(data, axis=1)
        per_vector_mins = np.min(data, axis=1)
        per_vector_maxs = np.max(data, axis=1)

        summary = {
            'shape': data.shape,
            'mean': {
                'per_vector': per_vector_means.tolist(),
                'global': np.mean(per_vector_means).tolist(),
                'percentiles': {
                    '1%': np.percentile(per_vector_means, 1).tolist(),
                    '10%': np.percentile(per_vector_means, 10).tolist(),
                    '25%': np.percentile(per_vector_means, 25).tolist(),
                    '50%': np.percentile(per_vector_means, 50).tolist(),
                    '75%': np.percentile(per_vector_means, 75).tolist(),
                    '90%': np.percentile(per_vector_means, 90).tolist(),
                    '99%': np.percentile(per_vector_means, 99).tolist(),
                }
            },
            'std': {
                'per_vector': per_vector_stds.tolist(),
                'global': np.mean(per_vector_stds).tolist(),
                'percentiles': {
                    '1%': np.percentile(per_vector_stds, 1).tolist(),
                    '10%': np.percentile(per_vector_stds, 10).tolist(),
                    '25%': np.percentile(per_vector_stds, 25).tolist(),
                    '50%': np.percentile(per_vector_stds, 50).tolist(),
                    '75%': np.percentile(per_vector_stds, 75).tolist(),
                    '90%': np.percentile(per_vector_stds, 90).tolist(),
                    '99%': np.percentile(per_vector_stds, 99).tolist(),
                }
            },
            'min': {
                'per_vector': per_vector_mins.tolist(),
                'global': np.mean(per_vector_mins).tolist(),
                'percentiles': {
                    '1%': np.percentile(per_vector_mins, 1).tolist(),
                    '10%': np.percentile(per_vector_mins, 10).tolist(),
                    '25%': np.percentile(per_vector_mins, 25).tolist(),
                    '50%': np.percentile(per_vector_mins, 50).tolist(),
                    '75%': np.percentile(per_vector_mins, 75).tolist(),
                    '90%': np.percentile(per_vector_mins, 90).tolist(),
                    '99%': np.percentile(per_vector_mins, 99).tolist(),
                }
            },
            'max': {
                'per_vector': per_vector_maxs.tolist(),
                'global': np.mean(per_vector_maxs).tolist(),
                'percentiles': {
                    '1%': np.percentile(per_vector_maxs, 1).tolist(),
                    '10%': np.percentile(per_vector_maxs, 10).tolist(),
                    '25%': np.percentile(per_vector_maxs, 25).tolist(),
                    '50%': np.percentile(per_vector_maxs, 50).tolist(),
                    '75%': np.percentile(per_vector_maxs, 75).tolist(),
                    '90%': np.percentile(per_vector_maxs, 90).tolist(),
                    '99%': np.percentile(per_vector_maxs, 99).tolist(),
                }
            }
        }
        with open(os.path.join(self.output_dir, f'{name}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    def save_encoded_embeddings_to_hdf5(self, encoded_embeddings, metadata, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('encoded_embeddings', data=encoded_embeddings)
            f.create_dataset('names', data=np.string_(metadata['names']))
            f.create_dataset('turn_numbers', data=metadata['turn_numbers'])
            f.create_dataset('file_paths', data=np.string_(metadata['file_paths']))
            f.create_dataset('model_names', data=np.string_(metadata['model_names']))

    def save_analysis_data(self, data, filename):
        np.save(filename, data)

    def visualize_embeddings(self, df, x_col, y_col, hue_col, title, file_name):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='tab20', s=5)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(self.output_dir, file_name))
        plt.close()

    def calculate_cluster_metrics(self, embeddings, clusters, true_labels):
        silhouette_avg = silhouette_score(embeddings, clusters)
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        return silhouette_avg, ari, nmi

    def calculate_all_metrics(self, embeddings_list, clusters_list, true_labels, n_jobs=4):
        results = Parallel(n_jobs=n_jobs)(delayed(self.calculate_cluster_metrics)(embeddings, clusters, true_labels) 
                                        for embeddings, clusters in zip(embeddings_list, clusters_list))
        return results

def main(embeddings_path, initial_size, increment_ratio, max_epochs, config, data_size=None):
    output_dir = f"{config['experiment_name']}-{datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertAutoencoder(device=device).to(device)
    trainer = Trainer(model, initial_size, increment_ratio, max_epochs)

    embeddings, metadata = trainer.load_embeddings(embeddings_path)
    if data_size:
        embeddings = embeddings[:data_size]
        for key in metadata:
            metadata[key] = metadata[key][:data_size]

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(embeddings_tensor)

    metrics = trainer.progressive_training(dataset)
    trainer.save_metrics(metrics, output_dir)

    # Save the model
    model_save_path = os.path.join(output_dir, 'trained_autoencoder.pth')
    torch.save(model.state_dict(), model_save_path)

    # Save the config
    config_save_path = os.path.join(output_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Load the model for encoding
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    analysis = Analysis(output_dir)
    
    # Encode embeddings using the autoencoder
    all_embeddings_encoded = []
    dataloader = DataLoader(dataset, batch_size=32768, shuffle=False, num_workers=4)
    
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
    analysis.summarize_data(embeddings, 'raw_embeddings')
    analysis.summarize_data(all_embeddings_encoded, 'encoded_embeddings')

    # Save encoded embeddings to an HDF5 file
    analysis.save_encoded_embeddings_to_hdf5(all_embeddings_encoded, metadata, os.path.join(output_dir, 'encoded_embeddings.h5'))

    # Dimensionality reduction with GPU UMAP for raw embeddings
    print("Reducing dimensions using UMAP with GPU for raw embeddings...")
    reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings_raw = reducer.fit_transform(embeddings)
    analysis.save_analysis_data(umap_embeddings_raw, os.path.join(output_dir, 'umap_embeddings_raw.npy'))

    # Perform KMeans clustering with GPU for raw embeddings
    print("Performing KMeans clustering with GPU for raw embeddings...")
    n_clusters = 10
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_raw.fit(umap_embeddings_raw)
    clusters_raw = kmeans_raw.labels_
    analysis.save_analysis_data(clusters_raw, os.path.join(output_dir, 'clusters_raw.npy'))

    # Dimensionality reduction with GPU UMAP for encoded embeddings
    print("Reducing dimensions using UMAP with GPU for encoded embeddings...")
    umap_embeddings_encoded = reducer.fit_transform(all_embeddings_encoded)
    analysis.save_analysis_data(umap_embeddings_encoded, os.path.join(output_dir, 'umap_embeddings_encoded.npy'))

    # Perform KMeans clustering with GPU for encoded embeddings
    print("Performing KMeans clustering with GPU for encoded embeddings...")
    kmeans_encoded = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_encoded.fit(umap_embeddings_encoded)
    clusters_encoded = kmeans_encoded.labels_
    analysis.save_analysis_data(clusters_encoded, os.path.join(output_dir, 'clusters_encoded.npy'))

    # PCA for raw embeddings
    print("Reducing dimensions using PCA for raw embeddings...")
    pca_raw = PCA(n_components=2)
    pca_embeddings_raw = pca_raw.fit_transform(embeddings)
    analysis.save_analysis_data(pca_embeddings_raw, os.path.join(output_dir, 'pca_embeddings_raw.npy'))

    # Perform KMeans clustering for PCA raw embeddings
    print("Performing KMeans clustering for PCA raw embeddings...")
    kmeans_pca_raw = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_pca_raw.fit(pca_embeddings_raw)
    clusters_pca_raw = kmeans_pca_raw.labels_
    analysis.save_analysis_data(clusters_pca_raw, os.path.join(output_dir, 'clusters_pca_raw.npy'))

    # PCA for encoded embeddings
    print("Reducing dimensions using PCA for encoded embeddings...")
    pca_encoded = PCA(n_components=2)
    pca_embeddings_encoded = pca_encoded.fit_transform(all_embeddings_encoded)
    analysis.save_analysis_data(pca_embeddings_encoded, os.path.join(output_dir, 'pca_embeddings_encoded.npy'))

    # Perform KMeans clustering for PCA encoded embeddings
    print("Performing KMeans clustering for PCA encoded embeddings...")
    kmeans_pca_encoded = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_pca_encoded.fit(pca_embeddings_encoded)
    clusters_pca_encoded = kmeans_pca_encoded.labels_
    analysis.save_analysis_data(clusters_pca_encoded, os.path.join(output_dir, 'clusters_pca_encoded.npy'))

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
    analysis.visualize_embeddings(df_raw, 'umap_x', 'umap_y', 'name', 'Raw UMAP Embeddings Colored by Names', 'raw_umap_colored_by_names.png')
    analysis.visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'name', 'Encoded UMAP Embeddings Colored by Names', 'encoded_umap_colored_by_names.png')

    # Visualize the UMAP embeddings with clusters for raw and encoded embeddings
    analysis.visualize_embeddings(df_raw, 'umap_x', 'umap_y', 'cluster', 'UMAP projection with KMeans clusters (Raw)', 'umap_kmeans_clusters_raw.png')
    analysis.visualize_embeddings(df_encoded, 'umap_x', 'umap_y', 'cluster', 'UMAP projection with KMeans clusters (Encoded)', 'umap_kmeans_clusters_encoded.png')

    # Visualize the PCA embeddings colored by names for raw and encoded embeddings
    analysis.visualize_embeddings(df_pca_raw, 'pca_x', 'pca_y', 'name', 'Raw PCA Embeddings Colored by Names', 'raw_pca_colored_by_names.png')
    analysis.visualize_embeddings(df_pca_encoded, 'pca_x', 'pca_y', 'name', 'Encoded PCA Embeddings Colored by Names', 'encoded_pca_colored_by_names.png')

    # Visualize the PCA embeddings with clusters for raw and encoded embeddings
    analysis.visualize_embeddings(df_pca_raw, 'pca_x', 'pca_y', 'cluster', 'PCA projection with KMeans clusters (Raw)', 'pca_kmeans_clusters_raw.png')
    analysis.visualize_embeddings(df_pca_encoded, 'pca_x', 'pca_y', 'cluster', 'PCA projection with KMeans clusters (Encoded)', 'pca_kmeans_clusters_encoded.png')

    # Calculate and print cluster metrics using parallel processing
    embeddings_list = [umap_embeddings_raw, umap_embeddings_encoded, pca_embeddings_raw, pca_embeddings_encoded]
    clusters_list = [clusters_raw, clusters_encoded, clusters_pca_raw, clusters_pca_encoded]
    
    results = analysis.calculate_all_metrics(embeddings_list, clusters_list, metadata['turn_numbers'], n_jobs=4)

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

    with open(os.path.join(output_dir, 'cluster_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Raw UMAP Embeddings: Silhouette Score = {results[0][0]}, ARI = {results[0][1]}, NMI = {results[0][2]}")
    print(f"Encoded UMAP Embeddings: Silhouette Score = {results[1][0]}, ARI = {results[1][1]}, NMI = {results[1][2]}")
    print(f"Raw PCA Embeddings: Silhouette Score = {results[2][0]}, ARI = {results[2][1]}, NMI = {results[2][2]}")
    print(f"Encoded PCA Embeddings: Silhouette Score = {results[3][0]}, ARI = {results[3][1]}, NMI = {results[3][2]}")

if __name__ == "__main__":
    embeddings_path = 'utterance_embeddings_ds1.h5'  # Path to the HDF5 file containing embeddings
    initial_size = 100  # Initial training data size and batch size
    increment_ratio = 0.2  # Increment ratio for progressive training
    max_epochs = 50000  # Maximum number of epochs

    config = {
        'experiment_name': 'high_accuracy_v1_visuals',
        'embeddings_path': embeddings_path,
        'model_path': 'trained_autoencoder.pth',
        'batch_size': 32768
    }

    data_size = 2000  # Set a specific value for quick testing, e.g., 10000

    main(embeddings_path, initial_size, increment_ratio, max_epochs, config, data_size)
