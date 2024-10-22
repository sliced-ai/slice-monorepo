import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Utility Function to Convert Grayscale to RGB
def grayscale_to_rgb(image):
    """
    Converts a single-channel (grayscale) image to a three-channel (RGB) image by repeating the channel.
    """
    return image.repeat(3, 1, 1)

# 2. Dataset Preparation for Multiple Datasets
def get_datasets(num_tasks):
    """
    Define a list of datasets for different tasks. 
    For simplicity, we'll use pre-defined datasets available in torchvision.
    """
    datasets_list = []
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize all images to 32x32 for consistency
        transforms.ToTensor(),
        transforms.Lambda(grayscale_to_rgb)  # Convert grayscale to RGB using the defined function
    ])

    # Task 1: MNIST (Grayscale)
    datasets_list.append(datasets.MNIST('./data', train=True, download=True, transform=transform_rgb))

    # Task 2: Fashion MNIST (Grayscale)
    datasets_list.append(datasets.FashionMNIST('./data', train=True, download=True, transform=transform_rgb))

    # Task 3: CIFAR-10 (RGB)
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    datasets_list.append(datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar))

    # Task 4: SVHN (RGB)
    transform_svhn = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    datasets_list.append(datasets.SVHN('./data', split='train', download=True, transform=transform_svhn))

    # Task 5: STL10 (RGB, Resized to 32x32 for consistency)
    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    datasets_list.append(datasets.STL10('./data', split='train', download=True, transform=transform_stl10))
    
    if len(datasets_list) < num_tasks:
        raise ValueError(f"Not enough predefined datasets for {num_tasks} tasks.")
    
    return datasets_list[:num_tasks]

def get_dataloaders(datasets_list, batch_size=64):
    """
    Create DataLoaders for the provided list of datasets.
    """
    dataloaders = []
    for dataset in datasets_list:
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    return dataloaders

# 3. Model Definition with Dynamic Input Channels and Per-Task Heads
class Net(nn.Module):
    def __init__(self, num_tasks, num_classes_per_task, input_channels=3):
        super(Net, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),  # Dynamic input channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),  # Adjust based on final dimensions after convolutions
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Task-specific heads
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.heads.append(nn.Linear(256, num_classes_per_task))
    
    def forward(self, x, task_id):
        x = self.shared(x)
        x = self.heads[task_id](x)
        return x

# 4. EWC Implementation
class EWC:
    def __init__(self, model: nn.Module, dataloader: DataLoader, task_id, device, lambda_=1000):
        self.model = copy.deepcopy(model)
        self.model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.task_id = task_id
        self.device = device
        self.lambda_ = lambda_
        # Store parameters of shared layers only
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if 'heads' not in n and p.requires_grad}
        self.fisher = self._fisher_information()
    
    def _fisher_information(self):
        """Estimate the Fisher Information Matrix for shared parameters."""
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if 'heads' not in n and p.requires_grad}
        self.model.zero_grad()
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            # Forward pass using the task-specific head
            output = self.model(data, self.task_id)
            loss = F.cross_entropy(output, target)
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if 'heads' not in n and p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
        # Average the Fisher Information
        for n in fisher:
            fisher[n] = fisher[n] / len(self.dataloader)
        return fisher
    
    def penalty(self, model: nn.Module):
        """Compute the EWC penalty for shared parameters."""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return self.lambda_ * loss

# 5. Training and Evaluation Functions
def train(model, optimizer, dataloader, ewc_list=None, writer=None, task_id=0, epoch=0):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, task_id)
        loss = F.cross_entropy(output, target)
        if ewc_list:
            ewc_loss = 0
            for ewc in ewc_list:
                ewc_loss += ewc.penalty(model)
            loss += ewc_loss
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
                output = model(data, idx)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
            acc = correct / total
            accuracies.append(acc)
            if writer:
                writer.add_scalar(f'Eval/Accuracy_Task_{idx+1}_After_Task_{task_id+1}', acc, epoch)
            print(f'Accuracy on Task {idx+1}: {acc*100:.2f}%')
    return accuracies

# 6. Main Training Loop with Multiple EWC Constraints and Mixed Dataset
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
    
    # Prepare datasets and dataloaders for multiple tasks
    datasets_list = get_datasets(num_tasks=args.num_tasks)
    dataloaders_list = get_dataloaders(datasets_list, batch_size=args.batch_size)
    
    # Initialize model with dynamic input channels
    input_channels = 3  # All datasets converted to 3 channels (RGB)
    num_classes_per_task = 10  # Assuming each task has 10 classes for simplicity
    model = Net(num_tasks=args.num_tasks, num_classes_per_task=num_classes_per_task, input_channels=input_channels).to(device)
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize list to store EWC instances
    ewc_list = []
    
    # Initialize results storage
    num_tasks = args.num_tasks
    results = np.zeros((num_tasks, num_tasks))
    
    # Storage for mixed dataset of all tasks learned so far
    all_data = []
    
    # Training loop
    for task in range(num_tasks):
        print(f'\n=== Training on Task {task+1}/{num_tasks} ===')
        
        # Combine current task data with all previously learned tasks
        if len(all_data) > 0:
            combined_dataset = ConcatDataset(all_data + [datasets_list[task]])
        else:
            combined_dataset = datasets_list[task]
        
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Train for specified epochs
        for epoch in range(1, args.epochs+1):
            train(model, optimizer, combined_loader, ewc_list, writer, task_id=task, epoch=epoch)
            evaluate(model, dataloaders_list[:task+1], task_id=task, writer=writer, epoch=epoch)
            scheduler.step()
        
        # After training on the current task, save EWC constraints
        if args.ewc_lambda > 0:
            ewc = EWC(model, dataloaders_list[task], task_id=task, device=device, lambda_=args.ewc_lambda)
            ewc_list.append(ewc)
        
        # Add current task data to the combined data list
        all_data.append(datasets_list[task])
        
        # Evaluate on all tasks up to current
        acc = evaluate(model, dataloaders_list[:task+1], task_id=task, writer=writer, epoch=args.epochs)
        results[task, :task+1] = acc
        
        # Save checkpoint (ensure all components are serializable)
        checkpoint_path = os.path.join(checkpoints_dir, f'model_task_{task+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ewc_list': [copy.deepcopy(ewc).params for ewc in ewc_list]  # Only save EWC params to avoid serialization issues
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')
    
    # Plot results
    plot_results(results, num_tasks, plots_dir, args.save_fig)
    writer.close()

# 7. Plotting Function
def plot_results(results, num_tasks, plots_dir, save_fig=False):
    plt.figure(figsize=(10, 8))
    for task in range(num_tasks):
        plt.plot(range(1, task+2), results[task, :task+1], label=f'Task {task+1}')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('EWC Continual Learning Performance on Multiple Datasets')
    plt.legend()
    plt.grid(True)
    if save_fig:
        plot_path = os.path.join(plots_dir, 'ewc_continual_learning_results.png')
        plt.savefig(plot_path)
        print(f'Results plot saved at {plot_path}')
    plt.show()

# 8. Argument Parser for Flexibility
def parse_args():
    parser = argparse.ArgumentParser(description='Advanced EWC Continual Learning with Multiple Datasets')
    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks (datasets) to use')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ewc_lambda', type=float, default=0, help='EWC regularization strength')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='results', help='Base directory to save all outputs')
    parser.add_argument('--save_fig', action='store_true', help='Save the results plot as an image')
    return parser.parse_args()

# 9. Entry Point
if __name__ == '__main__':
    args = parse_args()
    main(args)
