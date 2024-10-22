import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Dataset Preparation
def get_split_mnist(num_tasks=5):
    """
    Split MNIST into `num_tasks` classification tasks.
    Each task has its own subset of classes.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    classes = mnist_train.targets.unique().tolist()
    classes_per_task = len(classes) // num_tasks
    train_loaders = []
    test_loaders = []
    task_classes = []

    for i in range(num_tasks):
        start = i * classes_per_task
        end = (i + 1) * classes_per_task if i < num_tasks -1 else len(classes)
        task_class = list(range(start, end))
        task_classes.append(task_class)
        
        # Train subset
        train_indices = [idx for idx, target in enumerate(mnist_train.targets) if target in task_class]
        train_subset = Subset(mnist_train, train_indices)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        train_loaders.append(train_loader)
        
        # Test subset
        test_indices = [idx for idx, target in enumerate(mnist_test.targets) if target in task_class]
        test_subset = Subset(mnist_test, test_indices)
        test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)
        test_loaders.append(test_loader)
        
    return train_loaders, test_loaders, task_classes

# 2. Model Definition with Single Output Head
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # 26x26 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 24x24 -> 12x12
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Single output layer for all tasks
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.shared(x)
        x = self.fc(x)
        return x

# 3. Training and Evaluation Functions
def train(model, optimizer, dataloader, writer=None, task_id=0, epoch=0):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        total_loss += loss.item()
        if writer and batch_idx % 100 == 0:
            writer.add_scalar(f'Train/Loss_Task_{task_id+1}_Epoch_{epoch}', loss.item(), epoch * len(dataloader) + batch_idx)
    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar(f'Train/Average_Loss_Task_{task_id+1}_Epoch_{epoch}', avg_loss, epoch)
    print(f'Train Task {task_id+1}, Epoch {epoch}, Loss: {avg_loss:.4f}')

def evaluate(model, dataloaders, task_id, writer=None, epoch=0):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for idx, dl in enumerate(dataloaders):
            correct = 0
            total = 0
            for data, target in dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
            acc = correct / total
            accuracies.append(acc)
            if writer:
                writer.add_scalar(f'Eval/Accuracy_Task_{idx+1}_After_Task_{task_id+1}', acc, epoch)
            print(f'Accuracy on Task {idx+1}: {acc*100:.2f}%')
    return accuracies

# 4. Main Training Loop for Standard Model
def main(args):
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, 'logs')
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)
    
    # Prepare datasets
    train_loaders, test_loaders, task_classes = get_split_mnist(num_tasks=args.num_tasks)
    
    # Initialize model
    model = Net(num_classes=10).to(device)
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize results storage
    num_tasks = args.num_tasks
    results = np.zeros((num_tasks, num_tasks))
    
    # Training loop
    for task in range(num_tasks):
        print(f'\n=== Training on Task {task+1}/{num_tasks} ===')
        
        # Train for specified epochs
        for epoch in range(1, args.epochs+1):
            train(model, optimizer, train_loaders[task], writer, task_id=task, epoch=epoch)
            evaluate(model, test_loaders[:task+1], task_id=task, writer=writer, epoch=epoch)
            scheduler.step()
        
        # Evaluate on all tasks up to current
        acc = evaluate(model, test_loaders[:task+1], task_id=task, writer=writer, epoch=args.epochs)
        results[task, :task+1] = acc
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f'model_task_{task+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')
    
    # Plot results
    plot_results(results, num_tasks, plots_dir, args.save_fig)
    writer.close()

# 5. Plotting Function
def plot_results(results, num_tasks, plots_dir, save_fig=False):
    plt.figure(figsize=(10, 8))
    for task in range(num_tasks):
        plt.plot(range(1, task+2), results[task, :task+1], label=f'Task {task+1}')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Standard Model Continual Learning Performance on Split MNIST')
    plt.legend()
    plt.grid(True)
    if save_fig:
        plot_path = os.path.join(plots_dir, 'standard_continual_learning_results.png')
        plt.savefig(plot_path)
        print(f'Results plot saved at {plot_path}')
    plt.show()

# 6. Argument Parser for Flexibility
def parse_args():
    parser = argparse.ArgumentParser(description='Standard Model Continual Learning on Split MNIST')
    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks to split MNIST into')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results_standard', help='Base directory to save all outputs')
    parser.add_argument('--save_fig', action='store_true', help='Save the results plot as an image')
    return parser.parse_args()

# 7. Entry Point
if __name__ == '__main__':
    args = parse_args()
    main(args)
