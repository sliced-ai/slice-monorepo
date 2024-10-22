import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import argparse
import os

# Set random seeds for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Dataset Preparation
def get_mnist_dataloaders(batch_size=64):
    """
    Load MNIST dataset and create DataLoaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 2. Model Definitions
class SimpleNet(nn.Module):
    """
    A simple neural network with one hidden layer.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 3. EWC Implementation
class EWC:
    """
    Elastic Weight Consolidation (EWC) to mitigate catastrophic forgetting.
    """
    def __init__(self, model: nn.Module, dataloader: DataLoader, device, lambda_=1000):
        self.model = copy.deepcopy(model)
        self.model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.device = device
        self.lambda_ = lambda_
        # Store model parameters
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        """
        Estimate the Fisher Information Matrix for the model parameters.
        """
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.zero_grad()
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
        # Average the Fisher Information
        for n in fisher:
            fisher[n] = fisher[n] / len(self.dataloader)
        return fisher

    def penalty(self, model: nn.Module):
        """
        Compute the EWC penalty for the current model.
        """
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return self.lambda_ * loss

# 4. Optimizer Model Definition
class OptimizerModel(nn.Module):
    """
    A neural network that takes the Student's loss as input and outputs a modified loss.
    """
    def __init__(self):
        super(OptimizerModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Bound output between -1 and 1 to prevent extreme loss modifications
        )

    def forward(self, loss_input):
        """
        Forward pass for the Optimizer model.
        Args:
            loss_input (torch.Tensor): Tensor containing the Student's loss (shape: [batch_size, 1]).
        Returns:
            torch.Tensor: Modified loss (shape: [batch_size, 1]).
        """
        modified_loss = self.fc(loss_input)
        return modified_loss

# Weight Initialization Function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 5. Training and Evaluation Functions
def train_teacher(model, optimizer, dataloader, ewc, device):
    """
    Train the Teacher model on the current batch.
    """
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        if ewc:
            loss += ewc.penalty(model)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Teacher Training Loss: {avg_loss:.4f}')
    return avg_loss

def compute_similarity_normalized(teacher_deltas, student_deltas):
    """
    Compute normalized cosine similarity between teacher and student parameter deltas.
    Args:
        teacher_deltas (torch.Tensor): Flattened parameter deltas of the teacher model.
        student_deltas (torch.Tensor): Flattened parameter deltas of the student model.
    Returns:
        torch.Tensor: Cosine similarity score.
    """
    if torch.norm(teacher_deltas) == 0 or torch.norm(student_deltas) == 0:
        return torch.tensor(0.0, device=teacher_deltas.device)
    similarity = F.cosine_similarity(teacher_deltas, student_deltas, dim=0)
    return similarity

def train_student(student_model, optimizer_model, dataloader, teacher_model, ewc, optimizer_model_optimizer, teacher_avg_loss, device, max_optimizer_steps=10):
    """
    Train the Student model using the Optimizer model to modify its loss.
    The goal is to maximize the similarity between the Student's updates and the Teacher's updates,
    while keeping the student's loss close to the teacher's average loss.
    """
    student_model.train()
    teacher_model.eval()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Step 1: Teacher Model Training and Loss Computation
        teacher_params_before = {n: p.clone().detach() for n, p in teacher_model.named_parameters()}

        # Perform teacher forward and backward passes to calculate gradients
        teacher_optimizer_step = optim.SGD(teacher_model.parameters(), lr=0.0005)  # Dummy optimizer
        teacher_optimizer_step.zero_grad()
        teacher_output = teacher_model(data)
        teacher_loss = F.cross_entropy(teacher_output, target)
        if ewc:
            teacher_loss += ewc.penalty(teacher_model)
        teacher_loss.backward()
        teacher_optimizer_step.step()

        # Compute teacher parameter deltas
        teacher_deltas = []
        for n, p in teacher_model.named_parameters():
            delta = p - teacher_params_before[n]
            teacher_deltas.append(delta.view(-1))
        teacher_deltas = torch.cat(teacher_deltas).to(device)

        # Step 2: Student Model and Optimizer Model Interaction
        student_params_before = {n: p.clone().detach() for n, p in student_model.named_parameters()}

        # Perform student forward pass and compute initial loss without backward pass
        student_output = student_model(data)
        student_loss = F.cross_entropy(student_output, target)
        initial_student_loss_value = student_loss.item()

        # Prepare loss input for Optimizer Model (shape: [1])
        loss_input = student_loss.view(1, -1)

        # Placeholder for optimal parameter updates
        best_similarity = float('-inf')
        best_updates = None

        # Optimization Loop: Maximizing Similarity
        for step in range(max_optimizer_steps):
            # Reset student model to its initial state before optimization steps
            temp_student_model = copy.deepcopy(student_model).to(device)  # Create a fresh student model copy

            # Detach tensors and re-evaluate student loss for a fresh computational graph
            temp_student_output = temp_student_model(data)
            temp_student_loss = F.cross_entropy(temp_student_output, target)

            # Step 2.2: Optimizer Model Forward Pass
            with torch.enable_grad():
                loss_input = temp_student_loss.view(1, -1)  # Ensure loss_input is part of the graph
                modified_loss_value = optimizer_model(loss_input)

            scale_factor = 0.7  # Scaling factor to prevent large loss modifications
            modified_loss = temp_student_loss * (modified_loss_value.squeeze() * scale_factor)

            # Step 2.3: Compute gradients of modified_loss w.r.t temp_student_model's parameters
            grads = torch.autograd.grad(modified_loss, temp_student_model.parameters(), create_graph=True, allow_unused=True)
            if None in grads:
                continue

            # Manually update temporary student model's parameters
            temp_updated_params = {}
            for (name, param), grad in zip(temp_student_model.named_parameters(), grads):
                temp_updated_params[name] = param - 0.05 * grad  # 0.05 is the learning rate

            # Compute student parameter deltas
            temp_student_deltas = []
            for name, param in temp_student_model.named_parameters():
                delta = temp_updated_params[name] - student_params_before[name]
                temp_student_deltas.append(delta.view(-1))
            temp_student_deltas = torch.cat(temp_student_deltas).to(device)

            # Compute similarity based on parameter deltas
            similarity_post = compute_similarity_normalized(teacher_deltas, temp_student_deltas)

            # Compute loss preservation penalty based on teacher's average loss
            loss_difference = (temp_student_loss - teacher_avg_loss).abs()  # Absolute difference from teacher's avg loss
            loss_preservation_penalty = loss_difference * 5  # Increase penalty weight if needed

            # Debugging: Print loss difference
            #print(f"Loss Diff: {loss_difference.item():.4f}, Penalty: {loss_preservation_penalty.item():.4f}")

            # Combined objective: Maximize similarity, minimize loss change
            combined_objective = -similarity_post #+ 0.1*loss_preservation_penalty

            # Backpropagate the combined objective to the optimizer model's parameters
            optimizer_model_optimizer.zero_grad()
            combined_objective.backward()
            optimizer_model_optimizer.step()

            # Update best similarity and best updates if current similarity is better
            if similarity_post.item() > best_similarity:
                best_similarity = similarity_post.item()
                best_updates = temp_updated_params

            # Print single-line metrics for each step
        print(f"S{step + 1}: ISL:{initial_student_loss_value:.4f} -> PSL:{temp_student_loss.item():.4f}, Sim:{similarity_post.item():.4f}, "
                  f"LPen:{loss_preservation_penalty.item():.4f}, ES:{combined_objective.item():.4f}")

        # Apply the best updates found during the optimization steps
        if best_updates is not None:
            with torch.no_grad():
                for name, param in student_model.named_parameters():
                    param.copy_(best_updates[name])

        # After optimization steps, compute final similarity
        final_similarity = best_similarity

        # Accumulate the modified loss for this batch
        total_loss += student_loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'\nStudent Training Loss (Modified): {avg_loss:.4f}\n')

    return avg_loss


def evaluate(model, dataloader, device, model_name='Model'):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f'{model_name} Accuracy: {accuracy*100:.2f}%')
    return accuracy

# 6. Main Function
def main(args):
    # Prepare data loaders
    train_loader, test_loader = get_mnist_dataloaders(batch_size=args.batch_size)

    # Initialize Teacher and Student models
    teacher_model = SimpleNet().to(device)
    student_model = SimpleNet().to(device)

    # Initialize Optimizer Model
    optimizer_model = OptimizerModel().to(device)
    optimizer_model.apply(init_weights)  # Apply Xavier initialization

    # Initialize Optimizer for Teacher
    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.0005)

    # Initialize Optimizer for Optimizer Model
    optimizer_model_optimizer = optim.Adam(optimizer_model.parameters(), lr=args.optimizer_lr)

    # Initialize EWC for Teacher
    if args.use_ewc:
        ewc = EWC(teacher_model, train_loader, device, lambda_=args.ewc_lambda)
    else:
        ewc = None

    # Create directory for saving models
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        # Step 1: Teacher Model Training and Loss Computation
        print('\n--- Training Teacher Model ---')
        teacher_avg_loss = train_teacher(teacher_model, teacher_optimizer, train_loader, ewc, device)

        # Step 2: Student Model and Optimizer Model Interaction
        print('\n--- Training Student Model with Optimizer ---')
        student_avg_loss = train_student(
            student_model=student_model,
            optimizer_model=optimizer_model,
            dataloader=train_loader,
            teacher_model=teacher_model,
            ewc=ewc,
            optimizer_model_optimizer=optimizer_model_optimizer,
            teacher_avg_loss = teacher_avg_loss,
            device=device,
            max_optimizer_steps=args.max_optimizer_steps
        )

        # Step 3: Evaluation
        print('\n--- Evaluating Models ---')
        teacher_accuracy = evaluate(teacher_model, test_loader, device, model_name='Teacher')
        student_accuracy = evaluate(student_model, test_loader, device, model_name='Student')

        # Save models
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            teacher_path = os.path.join(args.save_dir, f'teacher_epoch_{epoch}.pth')
            student_path = os.path.join(args.save_dir, f'student_epoch_{epoch}.pth')
            optimizer_path = os.path.join(args.save_dir, f'optimizer_epoch_{epoch}.pth')
            torch.save(teacher_model.state_dict(), teacher_path)
            torch.save(student_model.state_dict(), student_path)
            torch.save(optimizer_model.state_dict(), optimizer_path)
            print(f'Saved Teacher, Student, and Optimizer models at epoch {epoch}.')

# 7. Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Teacher-Student Framework with Optimizer Model to Mimic EWC')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for Student model')
    parser.add_argument('--optimizer_lr', type=float, default=0.0001, help='Learning rate for Optimizer model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--use_ewc', action='store_true', help='Use EWC for Teacher model')
    parser.add_argument('--ewc_lambda', type=float, default=1000, help='EWC regularization strength')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save trained models')
    parser.add_argument('--save_interval', type=int, default=1, help='Epoch interval to save models')
    parser.add_argument('--max_optimizer_steps', type=int, default=5, help='Maximum number of optimizer steps per batch')
    return parser.parse_args()

# 8. Entry Point
if __name__ == '__main__':
    args = parse_args()
    main(args)
