import torch
import torch.nn as nn
import torch.optim as optim

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

### AutoEncoderTrainer Class

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

    def train_autoencoder(self, all_embeddings, epochs=50):
        self.model.train()
        all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32).to(self.device)
        for epoch in range(epochs):
            encoded, decoded = self.model(all_embeddings)
            loss = self.criterion(decoded, all_embeddings)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        combined_embedding = encoded.mean(dim=0)  # Example of combining embeddings
        return combined_embedding.detach().cpu().numpy(), self.model.state_dict()

### Test Case

def main():
    # Configurations
    encoder_config = {
        'input_size': 1000,  # Example input size
        'hidden_size': 512,  # Example hidden size
        'learning_rate': 0.001
    }

    # Create trainer
    trainer = AutoEncoderTrainer(encoder_config)

    # Example embeddings (simulating embeddings with values between 1000 and 4000)
    all_embeddings = [torch.randint(1000, 4001, (1000,)).numpy() for _ in range(10)]

    # Train autoencoder
    combined_embedding, model_weights = trainer.train_autoencoder(all_embeddings)

    # Output results
    print("Combined Embedding:", combined_embedding)
    print("Autoencoder Weights:", model_weights)

if __name__ == "__main__":
    main()
