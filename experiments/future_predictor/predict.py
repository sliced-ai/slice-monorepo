import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import copy

# Configurable parameters
DATASET_PATH = './data/CIFAR10'
SAVE_PATH = './model_data'
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
        x1 = torch.relu(self.conv1(x))
        x2 = self.pool(x1)
        x3 = torch.relu(self.conv2(x2))
        x4 = self.pool(x3)
        x5 = x4.view(x4.size(0), -1)  # Flatten the tensor
        x6 = torch.relu(self.fc1(x5))
        x7 = self.fc2(x6)
        return x7, [x1, x2, x3, x5, x6]

# Define the larger model for predicting activations
class LargerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LargerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom MSE loss for multiple layers' activations
def custom_mse_loss(pred_activations, true_activations):
    loss = 0
    for pred, true in zip(pred_activations, true_activations):
        loss += nn.MSELoss()(pred, true)
    return loss / len(pred_activations)

# Function to load activations and gradients
def load_activations_and_gradients(epoch):
    activations = torch.load(os.path.join(SAVE_PATH, f'activations_epoch_{epoch}.pth'))
    gradients = torch.load(os.path.join(SAVE_PATH, f'gradients_epoch_{epoch}.pth'))
    return activations, gradients

# Function to get final activations of all specified layers
def get_all_layer_activations(model, x):
    model.eval()
    with torch.no_grad():
        _, activations = model(x)
    return activations

# Training the larger model to predict next epoch activations
def train_larger_model_infinitely():
    print(f"Using device: {DEVICE}")

    # Load the smaller model
    smaller_model = SimpleCNN().to(DEVICE)

    # Load activations of epoch 0 and epoch 1
    activations_epoch_0, _ = load_activations_and_gradients(0)
    activations_epoch_1, _ = load_activations_and_gradients(1)

    # Assume single batch per epoch, extract the activations for the single batch
    activations_epoch_0 = activations_epoch_0[0]
    activations_epoch_1 = activations_epoch_1[0]

    # Get the final activations for all specified layers from epoch 0
    input_activations_list = get_all_layer_activations(smaller_model, activations_epoch_0[0].to(DEVICE))
    input_activations = torch.cat([act.flatten() for act in input_activations_list], dim=0).unsqueeze(0).to(DEVICE)

    # Get the activations for epoch 1 as targets (they should be many layers)
    target_activations_list = get_all_layer_activations(smaller_model, activations_epoch_1[0].to(DEVICE))
    target_activations = [act.to(DEVICE) for act in target_activations_list]

    # Print out the shapes of the activations
    print("Shape of input_activations:", input_activations.shape)
    for i, act in enumerate(target_activations):
        print(f"Shape of target_activations layer {i}:", act.shape)

    # Create the larger model
    input_size = input_activations.shape[1]
    output_size = sum(act.numel() for act in target_activations_list)
    larger_model = LargerModel(input_size, output_size).to(DEVICE)
    optimizer = optim.Adam(larger_model.parameters(), lr=LEARNING_RATE)

    # Set the initial state of the larger model
    model_state = torch.load(os.path.join(SAVE_PATH, f'model_epoch_0.pth'))
    larger_model.load_state_dict(model_state['model_state'])

    # Training loop
    step = 0
    while True:
        larger_model.train()
        optimizer.zero_grad()
        output = larger_model(input_activations)  # Get the prediction from the larger model

        # Reshape the output to match the expected shape of the smaller model's layers
        start = 0
        pred_activations = []
        for act in target_activations_list:
            end = start + act.numel()
            pred_activations.append(output[:, start:end].view(act.shape))
            start = end

        # Compute the loss
        loss = custom_mse_loss(pred_activations, target_activations)

        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        print(f"Step {step}: Loss = {loss.item():.4f}")
        step += 1

if __name__ == "__main__":
    train_larger_model_infinitely()
