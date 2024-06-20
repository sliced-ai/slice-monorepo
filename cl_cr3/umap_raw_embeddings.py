# Install necessary packages (skip this if running in a RAPIDS environment)
# !pip install pandas matplotlib seaborn h5py
# RAPIDS.ai libraries are installed via conda as shown earlier

# Import required libraries
import numpy as np
import pandas as pd
from cuml.manifold import UMAP
from cuml.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os

# Function to load embeddings from an HDF5 file with proper decoding
def load_embeddings_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        embeddings = np.array(f['embeddings'])
        names = [name.decode('utf8') for name in f['names']]
        utterances = [utt.decode('utf8') for utt in f['utterances']]
        turn_numbers = np.array(f['turn_numbers'])
    
    return embeddings, names, utterances, turn_numbers

# Function to load UMAP embeddings from an HDF5 file
def load_umap_embeddings_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        umap_embeddings = np.array(f['umap_embeddings'])
        clusters = np.array(f['clusters'])
    return umap_embeddings, clusters

# Function to reduce dimensions using UMAP with GPU support
def umap_gpu(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

# Function to save UMAP embeddings to an HDF5 file
def save_umap_embeddings_to_hdf5(umap_embeddings, clusters, df, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('umap_embeddings', data=umap_embeddings)
        f.create_dataset('names', data=np.string_(df['name'].values))
        f.create_dataset('utterances', data=np.string_(df['utterance'].values))
        f.create_dataset('turn_numbers', data=df['turn_number'].values)
        f.create_dataset('clusters', data=clusters)

# Main function to execute the script
def main():
    # Parameters
    embeddings_file = 'utterance_embeddings.h5'
    umap_file = 'umap_embeddings.h5'
    use_preprocessed_umap = True  # Flag to use preprocessed UMAP embeddings if available
    
    # Load embeddings from HDF5 file
    embeddings, names, utterances, turn_numbers = load_embeddings_from_hdf5(embeddings_file)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame({
        'name': names,
        'utterance': utterances,
        'turn_number': turn_numbers
    })
    
    if use_preprocessed_umap and os.path.exists(umap_file):
        print("Loading preprocessed UMAP embeddings from file...")
        umap_embeddings, clusters = load_umap_embeddings_from_hdf5(umap_file)
        print("UMAP embeddings loaded from file.")
    else:
        # Reduce dimensions using UMAP with GPU
        print("Reducing dimensions using UMAP with GPU...")
        umap_embeddings = umap_gpu(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine')
        
        # Perform KMeans clustering with GPU
        print("Performing KMeans clustering with GPU...")
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(umap_embeddings)
        clusters = kmeans.labels_
        
        # Save UMAP embeddings to an HDF5 file
        print("Saving UMAP embeddings to HDF5 file...")
        save_umap_embeddings_to_hdf5(umap_embeddings, clusters, df, umap_file)
    
    # Add UMAP and cluster labels to the DataFrame
    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]
    df['cluster'] = clusters
    
    # Visualize the raw UMAP embeddings
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='umap_x', y='umap_y', data=df, s=5)
    plt.title('Raw UMAP Embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig('raw_umap_embeddings.png')
    plt.show()
    
    # Visualize the raw UMAP embeddings colored by names
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='umap_x', y='umap_y', hue='name', data=df, palette='tab20', s=5)
    plt.title('Raw UMAP Embeddings Colored by Names')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('raw_umap_colored_by_names.png')
    plt.show()
    
    # Visualize the UMAP embeddings with clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='umap_x', y='umap_y', hue='cluster', data=df, palette='viridis', s=5)
    plt.title('UMAP projection with KMeans clusters')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Cluster')
    plt.savefig('umap_kmeans_clusters.png')
    plt.show()
    
    # Visualize the UMAP embeddings colored by names
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='umap_x', y='umap_y', hue='name', data=df, palette='tab20', s=5)
    plt.title('UMAP projection colored by names')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('umap_colored_by_names.png')
    plt.show()

if __name__ == '__main__':
    main()
