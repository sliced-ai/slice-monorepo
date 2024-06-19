import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from cuml.manifold import UMAP as cumlUMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        embeddings = f['embeddings'][:]
        names = f['names'][:]
    
    metadata = {
        'names': [name.decode('utf8') for name in names],
    }
    return embeddings, metadata

def compute_statistics(embeddings):
    df = pd.DataFrame(embeddings)
    stats = df.describe()
    print("Embedding Statistics:")
    print(stats)

def visualize_embeddings(embeddings, labels, title='Embeddings', file_name='embeddings.png'):
    plt.figure(figsize=(10, 7))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for idx, label in enumerate(unique_labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors(idx), label=label, alpha=0.5)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1), title='Names')
    plt.savefig(file_name)
    plt.show()

def main():
    embeddings_path = 'utterance_embeddings.h5'  # Path to the HDF5 file containing embeddings
    
    # Load embeddings and metadata
    embeddings, metadata = load_embeddings(embeddings_path)
    names = metadata['names']

    # Compute and print statistics
    compute_statistics(embeddings)

    # Dimensionality reduction with GPU UMAP
    reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(embeddings)

    # Visualize and save UMAP embeddings
    visualize_embeddings(umap_embeddings, names, 'UMAP Embeddings', 'umap_embeddings.png')

if __name__ == "__main__":
    main()
