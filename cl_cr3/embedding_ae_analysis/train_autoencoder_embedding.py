import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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

def calculate_cosine_similarity(outputs, inputs):
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

def prepare_dataloader(train_dataset, current_size, batch_size, num_workers=4):
    current_train_indices = torch.randperm(len(train_dataset))[:int(current_size)]
    current_train_dataset = Subset(train_dataset, current_train_indices)
    train_loader = DataLoader(current_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return train_loader

def progressive_training(model, train_dataset, initial_size, increment_ratio, max_epochs, device):
    total_size = len(train_dataset)
    test_size = 0.1
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    metrics = []

    # Separate the test data and ensure it's shuffled
    train_size = int((1 - test_size) * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=initial_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    current_size = initial_size

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_train_loader = executor.submit(prepare_dataloader, train_dataset, current_size, initial_size, num_workers=8)
        
        for epoch in range(max_epochs):
            if current_size > train_size:
                break

            train_loader = future_train_loader.result()
            prefetcher = Prefetcher(train_loader)
            future_train_loader = executor.submit(prepare_dataloader, train_dataset, current_size * (1 + increment_ratio), initial_size, num_workers=8)

            model.train()
            train_loss, train_similarity = 0, 0

            input = prefetcher.next()
            while input is not None:
                optimizer.zero_grad()
                outputs = model(input)
                loss = criterion(outputs, input)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_similarity += calculate_cosine_similarity(outputs, input)
                input = prefetcher.next()

            train_loss /= len(train_loader)
            train_similarity /= len(train_loader)

            model.eval()
            val_loss, val_similarity = 0, 0
            with torch.no_grad():
                for inputs in test_loader:
                    inputs = inputs[0].to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()
                    val_similarity += calculate_cosine_similarity(outputs, inputs)

            val_loss /= len(test_loader)
            val_similarity /= len(test_loader)

            metrics.append((current_size, train_loss, val_loss, train_similarity, val_similarity))
            current_size += current_size * increment_ratio

            print(f'Data size: {int(current_size)} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Train Similarity: {train_similarity:.2f}% | Val Similarity: {val_similarity:.2f}%')
    
    return metrics

def plot_metrics(metrics, save_path):
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

def main(embeddings_path, initial_size, increment_ratio, max_epochs, config):
    embeddings, metadata = load_embeddings(embeddings_path)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    dataset = TensorDataset(embeddings_tensor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertAutoencoder('bert-base-uncased', embedding_dim=1536).to(device)

    metrics = progressive_training(model, dataset, initial_size, increment_ratio, max_epochs, device)

    # Create output directory
    experiment_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = f"{config['experiment_name']}-{experiment_date}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    model_save_path = os.path.join(output_dir, 'trained_autoencoder.pth')
    torch.save(model.state_dict(), model_save_path)

    # Save the metrics plot
    plot_metrics(metrics, output_dir)

    # Save the config
    config_save_path = os.path.join(output_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    embeddings_path = 'utterance_embeddings.h5'  # Path to the HDF5 file containing embeddings
    initial_size = 4096  # Initial training data size and batch size
    increment_ratio = 0.005  # Increment ratio for progressive training
    max_epochs = 50000  # Maximum number of epochs

    config = {
        'experiment_name': 'bert_autoencoder_experiment'
    }

    main(embeddings_path, initial_size, increment_ratio, max_epochs, config)
