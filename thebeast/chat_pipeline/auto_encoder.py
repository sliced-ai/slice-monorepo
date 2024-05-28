import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy.ndimage import gaussian_filter
import os

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

    def train_autoencoder(self, all_embeddings, epochs=50):
        padded_embeddings = self.pad_embeddings(all_embeddings, self.encoder_config['input_size']).to(self.device)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            encoded, decoded = self.model(padded_embeddings)
            loss = self.criterion(decoded, padded_embeddings)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        combined_embedding = encoded.mean(dim=0)
        return combined_embedding.detach().cpu().numpy(), encoded.detach().cpu().numpy(), self.model.state_dict()

    def visualize_embeddings_tsne(self, embeddings, title='2D Visualization of Embeddings'):
        n_samples = len(embeddings)
        perplexity = min(40, n_samples - 1)  # Set perplexity to a value less than n_samples

        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, max_iter=300)
        tsne_results = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig('embedding_visualization_tsne.png')
        plt.show()

    def visualize_2d_grid(self, embeddings, grid_size=50, title='3D Visualization of Smoothed Text Embeddings'):
        embedding_size = embeddings.shape[1]
        adjusted_grid_size = int(np.sqrt(embedding_size))
        
        if adjusted_grid_size ** 2 != embedding_size:
            embeddings = [np.pad(embed, (0, adjusted_grid_size ** 2 - embed.size), 'constant') for embed in embeddings]
        
        smoothed_grids = np.array([gaussian_filter(embedding.reshape(adjusted_grid_size, adjusted_grid_size), sigma=2) for embedding in embeddings])

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(range(adjusted_grid_size), range(adjusted_grid_size))
        
        offset = 0.0
        for i in range(smoothed_grids.shape[0]):
            Z = smoothed_grids[i] + i * offset  # Offset each grid
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Embedding Value')

        plt.savefig('embedding_visualization_3d.png')
        plt.show()

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
