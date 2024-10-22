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
    'dropout': 0.1,               # Dropout rate
    'batch_size': 1,              # Batch size
    'num_epochs': 1,              # Number of training epochs
    'learning_rate': 1e-4,        # Learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': 'detokenized_output.json',  # Path to your data
    'clip_grad_norm': 1.0,         # Gradient clipping norm
    'save_model': False,           # Whether to save the model after training
    'model_save_path': 'standard_transformer.pt',  # Path to save the model
}

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
# Transformer Decoder Layer
# ---------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
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
        attn_output, attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # Feedforward network
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
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

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=config['max_seq_length'], dropout=config['dropout'])

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout']
            )
            for _ in range(config['num_layers'])
        ])

        # Final linear layer to map to vocabulary
        self.linear_out = nn.Linear(self.d_model, self.vocab_size)

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
