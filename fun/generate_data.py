import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
from tqdm import tqdm

# Configurable parameters
DATASET_PATH = './data/CIFAR10'
SAVE_PATH = './model_data'
EPOCHS = 10
BATCH_SIZE = 65536
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Define a simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to save model state
def save_model_state(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, path)

# Function to get activations of all layers
def get_activations(model, x):
    activations = []
    hooks = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook))

    model(x)

    for hook in hooks:
        hook.remove()

    return activations

# Training setup
def train_model():
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8)

    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_activations = []
        epoch_gradients = []
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Collect activations and gradients
            activations = get_activations(model, data)
            gradients = [param.grad.clone().cpu() for param in model.parameters()]
            epoch_activations.append(activations)
            epoch_gradients.append(gradients)

            optimizer.step()

            # Calculate loss and accuracy
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # Save model state and activations
        save_model_state(model, optimizer, epoch, os.path.join(SAVE_PATH, f'model_epoch_{epoch}.pth'))
        torch.save(epoch_activations, os.path.join(SAVE_PATH, f'activations_epoch_{epoch}.pth'))
        torch.save(epoch_gradients, os.path.join(SAVE_PATH, f'gradients_epoch_{epoch}.pth'))

if __name__ == "__main__":
    train_model()
