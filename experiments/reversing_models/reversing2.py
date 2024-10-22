import os

# Limit to 20 CPUs
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

import torch
torch.set_num_threads(20)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Iris dataset
iris = load_iris()
features, target = iris.data, iris.target

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
target = encoder.fit_transform(target.reshape(-1, 1))

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)

# Split into train and validation sets
features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize weights and biases manually
input_size = 4
hidden_layer_size = 100
output_size = 3

# Original model weights and biases
weights_layer1 = torch.randn(input_size, hidden_layer_size, requires_grad=True) * 0.01
biases_layer1 = torch.zeros(hidden_layer_size, requires_grad=True)
weights_layer2 = torch.randn(hidden_layer_size, output_size, requires_grad=True) * 0.01
biases_layer2 = torch.zeros(output_size, requires_grad=True)

# Reverse model weights and biases (will be transposed later)
rev_weights_layer2 = torch.zeros(output_size, hidden_layer_size, requires_grad=True)  # Reverse weights transposed
rev_biases_layer2 = torch.zeros(hidden_layer_size, requires_grad=True)
rev_weights_layer1 = torch.zeros(hidden_layer_size, input_size, requires_grad=True)  # Reverse weights transposed
rev_biases_layer1 = torch.zeros(input_size, requires_grad=True)

# Initialize Adam optimizer variables
momentum_weights_layer1 = torch.zeros_like(weights_layer1)
velocity_weights_layer1 = torch.zeros_like(weights_layer1)
momentum_biases_layer1 = torch.zeros_like(biases_layer1)
velocity_biases_layer1 = torch.zeros_like(biases_layer1)

momentum_weights_layer2 = torch.zeros_like(weights_layer2)
velocity_weights_layer2 = torch.zeros_like(weights_layer2)
momentum_biases_layer2 = torch.zeros_like(biases_layer2)
velocity_biases_layer2 = torch.zeros_like(biases_layer2)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Custom softmax function
def softmax(logits):
    exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

# Custom cross-entropy loss
def cross_entropy_loss(predicted_output, true_output):
    epsilon = 1e-8
    batch_size = true_output.shape[0]
    log_predictions = -torch.log(predicted_output + epsilon)
    loss = torch.sum(true_output * log_predictions) / batch_size
    return loss

# Forward pass for the normal model
def forward_pass_normal(features, weights_layer1, biases_layer1, weights_layer2, biases_layer2):
    hidden_layer_input = features.mm(weights_layer1) + biases_layer1
    hidden_layer_output = torch.relu(hidden_layer_input)
    output_layer_input = hidden_layer_output.mm(weights_layer2) + biases_layer2
    predicted_output = softmax(output_layer_input)
    return predicted_output, hidden_layer_output, output_layer_input  # Return logits

# Forward pass for the reverse model (takes hidden layer output from the original model as input)
def forward_pass_reverse(hidden_output, rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2):
    # Reverse the layers logically; take hidden layer output from original and reverse
    hidden_layer_input = hidden_output.mm(rev_weights_layer2) + rev_biases_layer2  # Reverse final layer becomes first
    input_layer_output = torch.relu(hidden_layer_input)
    output_layer_input = input_layer_output.mm(rev_weights_layer1) + rev_biases_layer1  # Reverse first layer becomes last
    return output_layer_input

# Backward pass
def backward_pass(features, predicted_output, true_output, hidden_layer_output, weights_layer2):
    batch_size = true_output.shape[0]
    
    # Gradients for output layer
    gradient_output_layer_input = (predicted_output - true_output) / batch_size
    gradient_weights_layer2 = hidden_layer_output.t().mm(gradient_output_layer_input)
    gradient_biases_layer2 = torch.sum(gradient_output_layer_input, dim=0)
    
    # Gradients for hidden layer
    gradient_hidden_layer_output = gradient_output_layer_input.mm(weights_layer2.t())
    gradient_hidden_layer_input = gradient_hidden_layer_output.clone()
    gradient_hidden_layer_input[hidden_layer_output <= 0] = 0  # derivative of ReLU
    gradient_weights_layer1 = features.t().mm(gradient_hidden_layer_input)
    gradient_biases_layer1 = torch.sum(gradient_hidden_layer_input, dim=0)
    
    return gradient_weights_layer1, gradient_biases_layer1, gradient_weights_layer2, gradient_biases_layer2

# Adam optimizer update
def adam_update(weights, biases, gradient_weights, gradient_biases, momentum_weights, velocity_weights, momentum_biases, velocity_biases, epoch, learning_rate, beta1, beta2, epsilon):
    # Update moment estimates
    momentum_weights = beta1 * momentum_weights + (1 - beta1) * gradient_weights
    velocity_weights = beta2 * velocity_weights + (1 - beta2) * (gradient_weights ** 2)
    momentum_biases = beta1 * momentum_biases + (1 - beta1) * gradient_biases
    velocity_biases = beta2 * velocity_biases + (1 - beta2) * (gradient_biases ** 2)
    
    # Bias correction
    momentum_weights_hat = momentum_weights / (1 - beta1 ** epoch)
    velocity_weights_hat = velocity_weights / (1 - beta2 ** epoch)
    momentum_biases_hat = momentum_biases / (1 - beta1 ** epoch)
    velocity_biases_hat = velocity_biases / (1 - beta2 ** epoch)
    
    # Update weights
    weights.data -= learning_rate * momentum_weights_hat / (torch.sqrt(velocity_weights_hat) + epsilon)
    biases.data -= learning_rate * momentum_biases_hat / (torch.sqrt(velocity_biases_hat) + epsilon)
    
    return momentum_weights, velocity_weights, momentum_biases, velocity_biases

# Transfer weights from the normal model to the reverse model (inverted form)
def transfer_weights_to_reverse(normal_weights_layer1, normal_biases_layer1, normal_weights_layer2, normal_biases_layer2):
    # Invert weights: the reverse model's first layer should be the original model's last
    rev_weights_layer1 = normal_weights_layer2.t().detach().requires_grad_(True)
    rev_biases_layer1 = normal_biases_layer2.detach().requires_grad_(True)
    rev_weights_layer2 = normal_weights_layer1.t().detach().requires_grad_(True)
    rev_biases_layer2 = normal_biases_layer1.detach().requires_grad_(True)
    
    return rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2

# Transfer weights back from the reverse model to the normal model
def transfer_weights_back_to_normal(rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2):
    # Invert weights back: the original model's last layer becomes the reverse model's first
    weights_layer2 = rev_weights_layer1.t().detach().requires_grad_(True)
    biases_layer2 = rev_biases_layer1.detach().requires_grad_(True)
    weights_layer1 = rev_weights_layer2.t().detach().requires_grad_(True)
    biases_layer1 = rev_biases_layer2.detach().requires_grad_(True)
    
    return weights_layer1, biases_layer1, weights_layer2, biases_layer2

# Training loop with two models (normal and reverse)
def train_with_reverse_model(features_train, target_train, features_val, target_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2, epochs, learning_rate):
    global momentum_weights_layer1, velocity_weights_layer1, momentum_biases_layer1, velocity_biases_layer1
    global momentum_weights_layer2, velocity_weights_layer2, momentum_biases_layer2, velocity_biases_layer2
    
    for epoch in range(epochs):
        # Step 1: Normal Forward pass (training) to capture logits
        predicted_output_train, hidden_layer_output_train, logits_train = forward_pass_normal(
            features_train, weights_layer1, biases_layer1, weights_layer2, biases_layer2)
        
        # Compute training loss
        loss_train = cross_entropy_loss(predicted_output_train, target_train)
        
        # Step 2: Backward pass (training)
        gradient_weights_layer1, gradient_biases_layer1, gradient_weights_layer2, gradient_biases_layer2 = backward_pass(
            features_train, predicted_output_train, target_train, hidden_layer_output_train, weights_layer2)
        
        # Perform Adam update
        epoch_count = epoch + 1
        momentum_weights_layer1, velocity_weights_layer1, momentum_biases_layer1, velocity_biases_layer1 = adam_update(
            weights_layer1, biases_layer1, gradient_weights_layer1, gradient_biases_layer1, 
            momentum_weights_layer1, velocity_weights_layer1, momentum_biases_layer1, velocity_biases_layer1, 
            epoch_count, learning_rate, beta1, beta2, epsilon)
        
        momentum_weights_layer2, velocity_weights_layer2, momentum_biases_layer2, velocity_biases_layer2 = adam_update(
            weights_layer2, biases_layer2, gradient_weights_layer2, gradient_biases_layer2, 
            momentum_weights_layer2, velocity_weights_layer2, momentum_biases_layer2, velocity_biases_layer2, 
            epoch_count, learning_rate, beta1, beta2, epsilon)
        
        # Step 3: Transfer weights to the reverse model (inverted)
        rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2 = transfer_weights_to_reverse(
            weights_layer1, biases_layer1, weights_layer2, biases_layer2)
        
        # Step 4: Reverse Forward pass (hidden layer output from normal model as input)
        rev_output_train = forward_pass_reverse(
            hidden_layer_output_train, rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2)
        
        # Compute reverse training loss (reverse forward pass reconstruction error)
        rev_loss_train = torch.mean((rev_output_train - features_train) ** 2)
        
        # Backward pass for reverse model (just like the normal one)
        rev_gradient_weights_layer1, rev_gradient_biases_layer1, rev_gradient_weights_layer2, rev_gradient_biases_layer2 = backward_pass(
            hidden_layer_output_train, rev_output_train, features_train, rev_output_train, rev_weights_layer2)
        
        # Perform Adam update for reverse model
        momentum_weights_layer1, velocity_weights_layer1, momentum_biases_layer1, velocity_biases_layer1 = adam_update(
            rev_weights_layer1, rev_biases_layer1, rev_gradient_weights_layer1, rev_gradient_biases_layer1, 
            momentum_weights_layer1, velocity_weights_layer1, momentum_biases_layer1, velocity_biases_layer1, 
            epoch_count, learning_rate, beta1, beta2, epsilon)
        
        momentum_weights_layer2, velocity_weights_layer2, momentum_biases_layer2, velocity_biases_layer2 = adam_update(
            rev_weights_layer2, rev_biases_layer2, rev_gradient_weights_layer2, rev_gradient_biases_layer2, 
            momentum_weights_layer2, velocity_weights_layer2, momentum_biases_layer2, velocity_biases_layer2, 
            epoch_count, learning_rate, beta1, beta2, epsilon)
        
        # Step 5: Transfer the updated weights from the reverse model back to the original model
        weights_layer1, biases_layer1, weights_layer2, biases_layer2 = transfer_weights_back_to_normal(
            rev_weights_layer1, rev_biases_layer1, rev_weights_layer2, rev_biases_layer2)

        # Step 6: Validation (forward pass with normal model)
        predicted_output_val, _, _ = forward_pass_normal(features_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2)
        
        # Compute validation loss
        loss_val = cross_entropy_loss(predicted_output_val, target_val)
        
        # Print loss and other information
        print(f'Epoch {epoch+1}/{epochs}, Loss (Train): {loss_train.item():.4f}, Reverse Loss (Train): {rev_loss_train.item():.4f}, Loss (Validation): {loss_val.item():.4f}')
        
# Hyperparameters
epochs = 1000
learning_rate = 0.01

# Train the network with two models (normal and reverse)
train_with_reverse_model(features_train, target_train, features_val, target_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2, epochs, learning_rate)
