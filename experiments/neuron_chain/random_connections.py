import torch
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Convert to PyTorch tensors and move to GPU
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights and biases manually
in_size = 4
hid_size = 1000
out_size = 3

w1 = torch.randn(in_size, hid_size, requires_grad=True, device=device) * 0.01
b1 = torch.zeros(hid_size, requires_grad=True, device=device)
w2 = torch.randn(hid_size, out_size, requires_grad=True, device=device) * 0.01
b2 = torch.zeros(out_size, requires_grad=True, device=device)

# Initialize Adam optimizer variables manually
m_w1 = torch.zeros_like(w1, device=device)
v_w1 = torch.zeros_like(w1, device=device)
m_b1 = torch.zeros_like(b1, device=device)
v_b1 = torch.zeros_like(b1, device=device)

m_w2 = torch.zeros_like(w2, device=device)
v_w2 = torch.zeros_like(w2, device=device)
m_b2 = torch.zeros_like(b2, device=device)
v_b2 = torch.zeros_like(b2, device=device)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize the active_connections list
active_connections = []

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

# Forward pass
def forward_pass(X, w1, b1, w2, b2, direct_connections=None):
    hidden_input = X.mm(w1) + b1
    hidden_output = torch.relu(hidden_input)
    
    # Apply direct connections if any
    if direct_connections:
        for (start_neuron, end_neuron) in direct_connections:
            hidden_output[:, end_neuron] += hidden_input[:, start_neuron]

    output_input = hidden_output.mm(w2) + b2
    pred_output = softmax(output_input)
    return pred_output, hidden_output

# Backward pass
def backward_pass(X, pred_output, true_output, hidden_output, w2):
    batch_size = true_output.shape[0]
    
    # Gradients for output layer
    grad_output_input = (pred_output - true_output) / batch_size
    grad_w2 = hidden_output.t().mm(grad_output_input)
    grad_b2 = torch.sum(grad_output_input, dim=0)
    
    # Gradients for hidden layer
    grad_hidden_output = grad_output_input.mm(w2.t())
    grad_hidden_input = grad_hidden_output.clone()
    grad_hidden_input[hidden_output <= 0] = 0  # derivative of ReLU
    grad_w1 = X.t().mm(grad_hidden_input)
    grad_b1 = torch.sum(grad_hidden_input, dim=0)
    
    return grad_w1, grad_b1, grad_w2, grad_b2

# Randomly add a direct connection between layers (Disabled for now)
def add_random_connection(hidden_size, connections):
    start_neuron = random.randint(0, hidden_size - 1)
    min_possible_end = start_neuron + 3  # min_steps
    max_possible_end = start_neuron + 10  # max_steps

    if min_possible_end >= hidden_size:
        min_possible_end = hidden_size - 1
    if max_possible_end >= hidden_size:
        max_possible_end = hidden_size - 1
    if min_possible_end > max_possible_end:
        min_possible_end = max_possible_end

    end_neuron = random.randint(min_possible_end, max_possible_end)
    
    connection = (start_neuron, end_neuron)
    connections.append(connection)
    return connections

# Randomly remove an existing connection (Disabled for now)
def remove_random_connection(connections):
    if connections:
        connection = random.choice(connections)
        connections.remove(connection)
    else:
        print("No connections to remove.")
    return connections
# Adam optimizer update manually
def adam_update(w, b, grad_w, grad_b, m_w, v_w, m_b, v_b, epoch, lr, beta1, beta2, epsilon):
    # Update moment estimates
    m_w = beta1 * m_w + (1 - beta1) * grad_w
    v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
    m_b = beta1 * m_b + (1 - beta1) * grad_b
    v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
    
    # Bias correction
    m_w_hat = m_w / (1 - beta1 ** epoch)
    v_w_hat = v_w / (1 - beta2 ** epoch)
    m_b_hat = m_b / (1 - beta1 ** epoch)
    v_b_hat = v_b / (1 - beta2 ** epoch)
    
    # Update weights
    w.data -= lr * m_w_hat / (torch.sqrt(v_w_hat) + epsilon)
    b.data -= lr * m_b_hat / (torch.sqrt(v_b_hat) + epsilon)
    
    return m_w, v_w, m_b, v_b

# Training loop with custom Adam optimizer and validation
def train(X_train, y_train, X_val, y_val, w1, b1, w2, b2, epochs, lr):
    global m_w1, v_w1, m_b1, v_b1
    global m_w2, v_w2, m_b2, v_b2
    global active_connections

    for epoch in range(epochs):
        # 0% chance to add or remove a connection for now
        if random.random() < 0.4:  # Set to 0.1 to enable adding connections
            active_connections = add_random_connection(hid_size, active_connections)
        
        if random.random() < 0.3:  # Set to 0.05 to enable removing connections
            active_connections = remove_random_connection(active_connections)
        
        # Forward pass (training)
        pred_train, hidden_train = forward_pass(X_train, w1, b1, w2, b2, active_connections)
        
        # Compute training loss
        loss_train = cross_entropy_loss(pred_train, y_train)
        
        # Backward pass (training)
        grad_w1, grad_b1, grad_w2, grad_b2 = backward_pass(X_train, pred_train, y_train, hidden_train, w2)

        # Perform custom Adam update
        epoch_count = epoch + 1
        m_w1, v_w1, m_b1, v_b1 = adam_update(
            w1, b1, grad_w1, grad_b1, m_w1, v_w1, m_b1, v_b1, epoch_count, lr, beta1, beta2, epsilon)
        
        m_w2, v_w2, m_b2, v_b2 = adam_update(
            w2, b2, grad_w2, grad_b2, m_w2, v_w2, m_b2, v_b2, epoch_count, lr, beta1, beta2, epsilon)

        # Forward pass (validation)
        pred_val, _ = forward_pass(X_val, w1, b1, w2, b2)
        
        # Compute validation loss
        loss_val = cross_entropy_loss(pred_val, y_val)
        
        # Print loss and other information
        print(f'Epoch {epoch+1}/{epochs}, Loss (Train): {loss_train.item():.4f}, Loss (Val): {loss_val.item():.4f}')

# Hyperparameters
epochs = 1000
lr = 0.01

# Train the network with custom Adam optimizer and validation
train(X_train, y_train, X_val, y_val, w1, b1, w2, b2, epochs, lr)
