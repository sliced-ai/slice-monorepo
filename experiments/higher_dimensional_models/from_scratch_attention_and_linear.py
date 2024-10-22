import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from transformers import GPT2TokenizerFast
import os

# ---------------------------
# Configuration Parameters
# ---------------------------
config = {
    'vocab_size': 50257,          # GPT-2 vocabulary size
    'max_seq_length': 64,         # Maximum sequence length
    'd_model': 256,               # Embedding dimension
    'nhead': 4,                   # Number of attention heads
    'dim_feedforward': 512,       # Feedforward network dimension
    'num_layers': 2,              # Number of decoder layers
    'num_comp_neurons': 5,        # Number of computation neurons in custom neuron
    'dropout': 0.1,               # Dropout rate
    'batch_size': 1,              # Batch size
    'num_epochs': 1,              # Number of training epochs
    'learning_rate': 1e-4,        # Learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': 'detokenized_output.json',  # Path to your data
    'clip_grad_norm': 1.0,         # Gradient clipping norm
    'save_model': False,           # Whether to save the model after training
    'model_save_path': 'custom_transformer_full.pt',  # Path to save the model
}

# ---------------------------
# Custom Neuron Module
# ---------------------------
class CustomNeuron(nn.Module):
    def __init__(self, in_features, out_features, num_comp_neurons=1, bias=True):
        super(CustomNeuron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_comp_neurons = num_comp_neurons
        self.bias = bias

        if self.num_comp_neurons < 1:
            raise ValueError("num_comp_neurons must be at least 1.")

        # Selection layer: decides which computation neuron to activate
        self.selection_layer = nn.Linear(in_features, out_features * self.num_comp_neurons, bias=bias)

        # Computation neurons: multiple linear transformations
        self.comp_weights = nn.Parameter(torch.Tensor(self.num_comp_neurons, out_features, in_features))
        if bias:
            self.comp_biases = nn.Parameter(torch.Tensor(self.num_comp_neurons, out_features))
        else:
            self.comp_biases = None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize computation neurons
        nn.init.xavier_uniform_(self.comp_weights)
        if self.comp_biases is not None:
            nn.init.zeros_(self.comp_biases)
        # Initialize selection layer
        nn.init.xavier_uniform_(self.selection_layer.weight)
        if self.selection_layer.bias is not None:
            nn.init.zeros_(self.selection_layer.bias)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, seq_len, in_features]
        Returns:
            output: Tensor of shape [batch_size, seq_len, out_features]
        """
        batch_size, seq_len, in_features = input.size()

        # Compute selection logits
        selection_logits = self.selection_layer(input)  # [b, s, o*n]
        selection_logits = selection_logits.view(batch_size, seq_len, self.out_features, self.num_comp_neurons)  # [b, s, o, n]

        # Compute selection probabilities using softmax over computation neurons
        selection_probs = F.softmax(selection_logits, dim=-1)  # [b, s, o, n]

        if self.num_comp_neurons == 1:
            # When num_comp_neurons=1, behave like a standard linear layer
            comp_output = torch.matmul(input, self.comp_weights[0].transpose(0, 1))  # [b, s, o]
            if self.comp_biases is not None:
                comp_output += self.comp_biases[0]  # [b, s, o]
            output = comp_output
        else:
            # Hard selection using argmax for forward pass
            with torch.no_grad():
                selected_idx = torch.argmax(selection_probs, dim=-1)  # [b, s, o]
                selected_mask_hard = F.one_hot(selected_idx, num_classes=self.num_comp_neurons).float()  # [b, s, o, n]

            # Allow gradients to flow through selection_probs
            # Create a soft mask for gradient flow
            selected_mask = selected_mask_hard - selection_probs.detach() + selection_probs  # [b, s, o, n]

            # Compute outputs from all computation neurons
            # [b, s, n, o] = [b, s, i] @ [n, o, i]
            comp_outputs = torch.einsum('bsi,noi->bsno', input, self.comp_weights)  # [b, s, n, o]

            # Add biases if present
            if self.comp_biases is not None:
                comp_biases_expanded = self.comp_biases.unsqueeze(0).unsqueeze(0)  # [1,1,n,o]
                comp_outputs = comp_outputs + comp_biases_expanded  # [b, s, n, o]

            # Apply selected_mask
            # Multiply comp_outputs with selected_mask and sum over n (computation neurons)
            # selected_mask: [b, s, o, n] -> [b, s, n, o]
            selected_mask_transposed = selected_mask.permute(0, 1, 3, 2)  # [b, s, n, o]
            output = torch.sum(comp_outputs * selected_mask_transposed, dim=2)  # [b, s, o]

        return output  # [b, s, o]

# ---------------------------
# Positional Encoding Module
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Not a parameter

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor after adding positional encoding and applying dropout
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ---------------------------
# Custom Multihead Attention Module
# ---------------------------
class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, num_comp_neurons, dropout=0.1):
        super(CustomMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_comp_neurons = num_comp_neurons

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_k = d_model // nhead

        # Replace Q, K, V linear layers with CustomNeuron
        self.q_linear = CustomNeuron(d_model, d_model, num_comp_neurons, bias=True)
        self.k_linear = CustomNeuron(d_model, d_model, num_comp_neurons, bias=True)
        self.v_linear = CustomNeuron(d_model, d_model, num_comp_neurons, bias=True)

        # Output linear layer
        self.out_proj = CustomNeuron(d_model, d_model, num_comp_neurons, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query, key, value: Tensors of shape [seq_len, batch_size, d_model]
            attn_mask: Tensor of shape [seq_len, seq_len]
        Returns:
            context: Tensor of shape [seq_len, batch_size, d_model]
        """
        # Transpose to [batch_size, seq_len, d_model]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        # Apply custom neurons
        Q = self.q_linear(query)  # [batch_size, seq_len, d_model]
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        batch_size, seq_len, _ = Q.size()
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, nhead, seq_len, seq_len]

        if attn_mask is not None:
            # attn_mask shape should be [seq_len, seq_len]
            # Expand mask to [batch_size, 1, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(0)  # [1, seq_len, seq_len]
            attn_mask = attn_mask.expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, nhead, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [batch_size, nhead, seq_len, d_k]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]

        # Apply output projection
        context = self.out_proj(context)  # [batch_size, seq_len, d_model]

        # Transpose back to [seq_len, batch_size, d_model]
        context = context.transpose(0, 1)

        return context

# ---------------------------
# Transformer Decoder Layer
# ---------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_comp_neurons, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, num_comp_neurons, dropout=dropout)

        # Feedforward network
        self.linear1 = CustomNeuron(d_model, dim_feedforward, num_comp_neurons, bias=True)
        self.linear2 = CustomNeuron(dim_feedforward, d_model, num_comp_neurons, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        """
        Args:
            tgt: Tensor of shape [seq_len, batch_size, d_model]
            tgt_mask: Tensor of shape [seq_len, seq_len]
        Returns:
            Tensor of shape [seq_len, batch_size, d_model]
        """
        # Self-attention
        attn_output = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # Feedforward network
        # Convert [seq_len, batch_size, d_model] to [batch_size, seq_len, d_model]
        tgt_transposed = tgt.transpose(0, 1)  # [batch_size, seq_len, d_model]
        ff_output = self.linear1(tgt_transposed)  # [batch_size, seq_len, dim_feedforward]
        ff_output = F.relu(ff_output)
        ff_output = self.linear2(ff_output)  # [batch_size, seq_len, d_model]
        ff_output = ff_output.transpose(0, 1)  # [seq_len, batch_size, d_model]
        tgt = tgt + self.dropout2(ff_output)
        tgt = self.norm2(tgt)

        return tgt

# ---------------------------
# Transformer Decoder Model
# ---------------------------
class TransformerDecoderModel(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderModel, self).__init__()
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.num_comp_neurons = config['num_comp_neurons']

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=config['max_seq_length'], dropout=config['dropout'])

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                num_comp_neurons=self.num_comp_neurons,
                dropout=config['dropout']
            )
            for _ in range(config['num_layers'])
        ])

        # Final linear layer to map to vocabulary
        self.linear_out = CustomNeuron(self.d_model, self.vocab_size, self.num_comp_neurons, bias=True)

    def forward(self, input_ids, tgt_mask=None):
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            tgt_mask: Tensor of shape [seq_len, seq_len]
        Returns:
            output: Tensor of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()

        # Embedding and positional encoding
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        embedded = self.positional_encoding(embedded)  # [batch_size, seq_len, d_model]

        # Transpose for transformer (seq_len, batch_size, d_model)
        x = embedded.transpose(0, 1)  # [seq_len, batch_size, d_model]

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, tgt_mask=tgt_mask)  # [seq_len, batch_size, d_model]

        # Transpose back to [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Final linear layer
        output = self.linear_out(x)  # [batch_size, seq_len, vocab_size]

        return output

    def generate_square_subsequent_mask(self, sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on the diagonal and below.
        Args:
            sz: Size of the mask (seq_len)
        Returns:
            mask: Tensor of shape [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.masked_fill(mask, float('-inf'))

# ---------------------------
# Data Preparation Function
# ---------------------------
def prepare_data(config, tokenizer):
    """
    Loads and tokenizes data from a JSON file.
    Args:
        config: Configuration dictionary
        tokenizer: Pretrained tokenizer
    Returns:
        data: List of tuples (tgt_input, tgt_output)
    """
    # Load data
    if not os.path.exists(config['data_path']):
        raise FileNotFoundError(f"Data file not found at {config['data_path']}")

    with open(config['data_path'], 'r', encoding='utf-8') as f:
        texts = json.load(f)

    data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=config['max_seq_length'], truncation=True)
        if len(tokens) < 2:
            continue  # Need at least two tokens for input and output
        input_ids = torch.tensor(tokens, dtype=torch.long)
        # For language modeling, target is input shifted by one
        tgt_input = input_ids[:-1]
        tgt_output = input_ids[1:]
        data.append((tgt_input, tgt_output))

    return data

# ---------------------------
# Training Loop Function
# ---------------------------
def train_model(model, config, data, tokenizer):
    """
    Trains the transformer model.
    Args:
        model: TransformerDecoderModel instance
        config: Configuration dictionary
        data: List of tuples (tgt_input, tgt_output)
        tokenizer: Pretrained tokenizer
    """
    model = model.to(config['device'])
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        for idx, (tgt_input, tgt_output) in enumerate(data):
            optimizer.zero_grad()

            # Add batch dimension
            tgt_input = tgt_input.unsqueeze(0).to(config['device'])  # [1, seq_len-1]
            tgt_output = tgt_output.unsqueeze(0).to(config['device'])  # [1, seq_len-1]

            # Create mask
            seq_len = tgt_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(config['device'])  # [seq_len, seq_len]

            # Forward pass
            outputs = model(tgt_input, tgt_mask=tgt_mask)  # [1, seq_len-1, vocab_size]

            # Reshape for loss
            outputs = outputs.view(-1, config['vocab_size'])  # [(1 * (seq_len-1)), vocab_size]
            tgt_output = tgt_output.view(-1)  # [(1 * (seq_len-1))]

            loss = criterion(outputs, tgt_output)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

            optimizer.step()

            total_loss += loss.item()

            # Print loss every 10 steps
            if (idx + 1) % 10 == 0:
                avg_loss = total_loss / (idx + 1)
                print(f"Epoch [{epoch +1}/{config['num_epochs']}], Step [{idx +1}/{len(data)}], Loss: {avg_loss:.4f}")

        # Average loss for the epoch
        avg_loss = total_loss / len(data)
        print(f"Epoch [{epoch +1}/{config['num_epochs']}], Average Loss: {avg_loss:.4f}")

        # Optionally save the model
        if config['save_model']:
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved to {config['model_save_path']}")

    print("Training Completed.")

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos_token to avoid padding issues

    # Prepare data
    data = prepare_data(config, tokenizer)
    print(f"Loaded {len(data)} training samples.")

    # Initialize model
    model = TransformerDecoderModel(config)
    print("Initialized TransformerDecoderModel.")

    # Train model
    train_model(model, config, data, tokenizer)

if __name__ == "__main__":
    main()
