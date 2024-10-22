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

weights_layer1 = torch.randn(input_size, hidden_layer_size, requires_grad=True) * 0.01
biases_layer1 = torch.zeros(hidden_layer_size, requires_grad=True)
weights_layer2 = torch.randn(hidden_layer_size, output_size, requires_grad=True) * 0.01
biases_layer2 = torch.zeros(output_size, requires_grad=True)

clip_value = 1.0  # Gradient clipping threshold

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

# Forward pass with logits
def forward_pass_with_logits(features, weights_layer1, biases_layer1, weights_layer2, biases_layer2):
    hidden_layer_input = features.mm(weights_layer1) + biases_layer1
    hidden_layer_output = torch.relu(hidden_layer_input)
    output_layer_input = hidden_layer_output.mm(weights_layer2) + biases_layer2
    predicted_output = softmax(output_layer_input)
    return predicted_output, hidden_layer_output, output_layer_input  # Return logits

# Backward pass with gradient clipping
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
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_([gradient_weights_layer1, gradient_biases_layer1, gradient_weights_layer2, gradient_biases_layer2], clip_value)
    
    return gradient_weights_layer1, gradient_biases_layer1, gradient_weights_layer2, gradient_biases_layer2

# Reverse pass
def reverse_pass(logits, weights_layer2, hidden_layer_output):
    gradient_output_layer_input = logits.mm(weights_layer2.t())
    gradient_hidden_layer_output = gradient_output_layer_input.clone()
    gradient_hidden_layer_output[hidden_layer_output <= 0] = 0  # ReLU derivative approximation
    return gradient_hidden_layer_output

# Compute error for reverse pass by mapping back to input space
def reverse_error(gradient_hidden_layer_output, weights_layer1, original_features):
    reconstructed_input = gradient_hidden_layer_output.mm(weights_layer1.t())
    return torch.mean((reconstructed_input - original_features) ** 2)

# SGD update
def sgd_update(weights, biases, gradient_weights, gradient_biases, learning_rate):
    # Update weights with SGD (basic update step)
    weights.data -= learning_rate * gradient_weights
    biases.data -= learning_rate * gradient_biases

# Training loop with SGD optimizer and reverse step every 5 epochs
def train_with_reverse_every_5_epochs(features_train, target_train, features_val, target_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2, epochs, learning_rate):
    for epoch in range(epochs):
        # Step 1: Forward pass (training) to capture logits
        predicted_output_train, hidden_layer_output_train, logits_train = forward_pass_with_logits(
            features_train, weights_layer1, biases_layer1, weights_layer2, biases_layer2)
        
        # Compute training loss
        loss_train = cross_entropy_loss(predicted_output_train, target_train)
        
        # Step 2: Backward pass (training)
        gradient_weights_layer1, gradient_biases_layer1, gradient_weights_layer2, gradient_biases_layer2 = backward_pass(
            features_train, predicted_output_train, target_train, hidden_layer_output_train, weights_layer2)
        
        # Perform SGD update
        sgd_update(weights_layer1, biases_layer1, gradient_weights_layer1, gradient_biases_layer1, learning_rate)
        sgd_update(weights_layer2, biases_layer2, gradient_weights_layer2, gradient_biases_layer2, learning_rate)

        # Perform reverse pass every 5 epochs
        if epoch % 100 == 0:
            # Step 3: Reverse inference from logits
            reverse_hidden_output = reverse_pass(logits_train, weights_layer2, hidden_layer_output_train)
            
            # Step 4: Compute error between reverse-propagated values and original input features
            reverse_loss = reverse_error(reverse_hidden_output, weights_layer1, features_train)
            
            # Step 5: Backpropagate this reverse loss as well
            reverse_gradient_weights_layer1 = features_train.t().mm(reverse_hidden_output)
            reverse_biases_layer1 = torch.sum(reverse_hidden_output, dim=0)
            
            # Clip gradients in the reverse pass
            torch.nn.utils.clip_grad_norm_([reverse_gradient_weights_layer1, reverse_biases_layer1], clip_value)
            
            # Perform a reverse update
            sgd_update(weights_layer1, biases_layer1, reverse_gradient_weights_layer1, reverse_biases_layer1, 0.001)

            print(f'Epoch {epoch+1}/{epochs}, Reverse Loss: {reverse_loss.item():.4f}')
        
        # Forward pass (validation)
        predicted_output_val, _, _ = forward_pass_with_logits(features_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2)
        
        # Compute validation loss
        loss_val = cross_entropy_loss(predicted_output_val, target_val)
        
        # Print loss and other information
        print(f'Epoch {epoch+1}/{epochs}, Loss (Train): {loss_train.item():.4f}, Loss (Validation): {loss_val.item():.4f}')
        
# Hyperparameters
epochs = 100000
learning_rate = 0.01

# Train the network with SGD optimizer and reverse step every 5 epochs
train_with_reverse_every_5_epochs(features_train, target_train, features_val, target_val, weights_layer1, biases_layer1, weights_layer2, biases_layer2, epochs, learning_rate)
