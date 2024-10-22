import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the Iris dataset
iris = load_iris()
features, target = iris.data, iris.target

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
target = encoder.fit_transform(target.reshape(-1, 1))

# Convert to PyTorch tensors and move to GPU
features = torch.tensor(features, dtype=torch.float32).to(device)
target = torch.tensor(target, dtype=torch.float32).to(device)

# Split into train and validation sets
features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Create DataLoader for batch processing
train_dataset = TensorDataset(features_train, target_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output, hidden

# Initialize the model, loss function, and optimizer
input_size = 4
hidden_layer_size = 100
output_size = 3

model = SimpleNN(input_size, hidden_layer_size, output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train(model, train_loader, features_val, target_val, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (features_batch, target_batch) in enumerate(train_loader):
            # Forward pass
            output, hidden = model(features_batch)
            
            # Compute loss
            loss = criterion(output, target_batch.argmax(dim=1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save hidden states
            hidden_states_df = pd.DataFrame(hidden.detach().cpu().numpy())
            hidden_states_df['Batch'] = batch_idx
            hidden_states_df['Epoch'] = epoch
            hidden_states_df.to_csv('hidden_states.csv', mode='a', header=False, index=False)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output, _ = model(features_val)
            val_loss = criterion(val_output, target_val.argmax(dim=1))
        
        # Print training and validation loss
        print(f'Epoch {epoch+1}/{epochs}, Loss (Train): {loss.item():.4f}, Loss (Validation): {val_loss.item():.4f}')
        model.train()

# Hyperparameters
epochs = 1000
learning_rate = 0.01

# Train the network
train(model, train_loader, features_val, target_val, epochs)
