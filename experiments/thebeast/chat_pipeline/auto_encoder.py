import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import json
import umap.umap_ as umap

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoderTrainer:
    def __init__(self, encoder_config):
        self.encoder_config = encoder_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(
            input_size=encoder_config['input_size'],
            hidden_size=encoder_config['hidden_size']
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=encoder_config['learning_rate'])

    def pad_embeddings(self, embeddings, target_length):
        padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
        return torch.narrow(padded_embeddings, 1, 0, target_length)

    def train_autoencoder(self, all_embeddings, epochs=5):
        padded_embeddings = self.pad_embeddings(all_embeddings, self.encoder_config['input_size']).to(self.device)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            encoded, decoded = self.model(padded_embeddings)
            loss = self.criterion(decoded, padded_embeddings)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
        combined_embedding = encoded.mean(dim=0)
        return combined_embedding.detach().cpu().numpy(), encoded.detach().cpu().numpy(), self.model.state_dict()


    def visualize_embeddings_tsne(self, embeddings, tsne_fig_path, json_path, title='2D Visualization of Embeddings'):
        n_neighbors = min(15, len(embeddings) - 1)  # Ensure n_neighbors is valid for the dataset size
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        try:
            umap_results = reducer.fit_transform(embeddings)
        except ValueError as e:
            print(f"UMAP error: {e}")
            # Handle potential zero-size array error or disconnected vertices
            umap_results = reducer.fit_transform(embeddings[:n_neighbors])  # Use a subset if necessary
    
        plt.figure(figsize=(10, 6))
        plt.scatter(umap_results[:, 0], umap_results[:, 1], alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig(tsne_fig_path)
        plt.close()
    
        # Save the UMAP results to a JSON file
        umap_data = {'embeddings': umap_results.tolist()}
        with open(json_path, 'w') as f:
            json.dump(umap_data, f, indent=4)



    def visualize_2d_grid(self, embeddings, grid_fig_path, json_path, grid_size=50, title='3D Visualization of Smoothed Text Embeddings'):
        embedding_size = embeddings.shape[1]
        adjusted_grid_size = int(np.sqrt(embedding_size))
        
        # Ensure all embeddings are reshaped to a square grid of the correct size
        if adjusted_grid_size ** 2 != embedding_size:
            embeddings = [np.pad(embed, (0, adjusted_grid_size ** 2 - embed.size), 'constant', constant_values=0) for embed in embeddings]
        
        # Smooth the grids
        smoothed_grids = np.array([gaussian_filter(embed.reshape(adjusted_grid_size, adjusted_grid_size), sigma=2) for embed in embeddings])
    
        # Plotting
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(range(adjusted_grid_size), range(adjusted_grid_size))
        
        # Adjust z-offset for each grid to avoid overlap
        offset = 0.0
        for i, grid in enumerate(smoothed_grids):
            Z = grid + i * offset  # Offset each grid slightly on the Z-axis
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
    
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Embedding Value')
        
        plt.savefig(grid_fig_path)  # Save directly to the provided path
        plt.close(fig)  # Close the figure to free up memory
        
        # Save the grid data to a JSON file
        grid_data = {'embeddings': smoothed_grids.tolist()}
        with open(json_path, 'w') as f:
            json.dump(grid_data, f, indent=4)

def main():
    encoder_config = {
        'input_size': 5000,
        'hidden_size': 512,
        'learning_rate': 0.001
    }

    trainer = AutoEncoderTrainer(encoder_config)

    all_embeddings = [
        torch.randint(1000, 4001, (3000,)).float(),
        torch.randint(1000, 4001, (1000,)).float(),
        torch.randint(1000, 4001, (5000,)).float()
    ]

    combined_embedding, embeddings, model_weights = trainer.train_autoencoder(all_embeddings)

    # Visualize embeddings using t-SNE
    trainer.visualize_embeddings_tsne(embeddings)

    # Visualize embeddings as a 2D grid
    trainer.visualize_2d_grid(embeddings)

if __name__ == "__main__":
    main()
