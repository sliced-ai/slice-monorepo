import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

# Configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "starting_learning_rate": 1e-5,
    "batch_size": 1,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "sequential_data_path": "/workspace/slice-monorepo/sub_validations/memory_training/tokenized_output_10k.json",
    "max_tokens": 4098,  # Maximum tokens in a batch
    "num_epochs": 1,  # Set to 1 for faster experimentation
    "experiment_dir": "/workspace/slice-monorepo/sub_validations/memory_training/experiments",
    "early_stop_step": 50  # Set this to stop training after a certain step and run validation early
}

# Special tokens
special_tokens = {
    "current_conv": "[CURRENT_CONV]",  # Replace with the actual special token used in the data
    "previous_conv": "[PREVIOUS_CONV]",  # Replace with the actual special token used in the data
    "eos_token": "<|endoftext|>",  # Use the tokenizer's EOS token
    "pad_token": "<|pad|>"  # Define a padding token if not already present in the tokenizer
}

# Dataset class for the tokenized_output_10k.json data
class SequentialMemoryPileDataset(Dataset):
    def __init__(self, token_file, tokenizer, max_len, start_idx=0, end_idx=None):
        with open(token_file, 'r') as f:
            self.tokens = json.load(f)[start_idx:end_idx]  # Load subset of JSON file for train/validation
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Ensure special tokens are added to the tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                special_tokens["current_conv"],
                special_tokens["previous_conv"],
                special_tokens["pad_token"]
            ]
        })

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = special_tokens["pad_token"]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Extract the tokenized sequence from the dataset
        token_data = self.tokens[idx]

        # Convert the list of tokens directly to tensors
        token_ids = torch.tensor(token_data, dtype=torch.long)

        # Convert special tokens to their corresponding IDs
        current_conv_id = self.tokenizer.convert_tokens_to_ids(special_tokens["current_conv"])
        previous_conv_id = self.tokenizer.convert_tokens_to_ids(special_tokens["previous_conv"])
        eos_token_id = self.tokenizer.convert_tokens_to_ids(special_tokens["eos_token"])

        # Find the positions of special tokens within the tokenized IDs
        current_conv_start = (token_ids == current_conv_id).nonzero(as_tuple=True)[0].item()
        previous_conv_start = (token_ids == previous_conv_id).nonzero(as_tuple=True)[0].item()

        # Extract `CURRENT_CONV` and `PREVIOUS_CONV` using the identified positions
        current_conv = token_ids[current_conv_start + 1: previous_conv_start]  # Between the current_conv and previous_conv tokens
        previous_conv = token_ids[previous_conv_start + 1:]  # After the previous_conv token

        # Insert the eos_token at the end of each conversation
        current_conv = torch.cat([current_conv, torch.tensor([eos_token_id], dtype=torch.long)])
        previous_conv = torch.cat([previous_conv, torch.tensor([eos_token_id], dtype=torch.long)])

        # Print the sizes of the AR (current_conv) and Seq2Seq (previous_conv) data segments
        print(f"AR (Current Conv) Size: {current_conv.size(0)}, Seq2Seq (Previous Conv) Size: {previous_conv.size(0)}")

        # Check if total combined size exceeds max_len
        combined_size = current_conv.size(0) + previous_conv.size(0)
        if combined_size > self.max_len:
            # Adjust by truncating the previous_conv sequence first
            excess_length = combined_size - self.max_len
            if previous_conv.size(0) > excess_length:
                previous_conv = previous_conv[:-excess_length]
            else:
                current_conv = current_conv[:self.max_len - previous_conv.size(0)]

        # Combine the sequences and pad to max_len
        combined = torch.cat([current_conv, previous_conv])
        padding = torch.full((self.max_len - combined.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
        combined_padded = torch.cat([combined, padding])

        # Update sizes after padding
        n = current_conv.size(0)
        k = previous_conv.size(0)

        return {
            'CURRENT_CONV': combined_padded[:n],
            'PREVIOUS_CONV': combined_padded[n:n + k],
            'PADDED_INPUT': combined_padded
        }

# Function to prepare attention masks correctly
def prepare_masks(n, k, total_len, device):
    # Create a basic attention mask where all tokens are visible
    mask = torch.ones((1, total_len), device=device)

    # Create a padding mask for the input
    padding_mask = torch.ones((1, total_len), device=device)

    # Ensure that padding tokens are masked
    padding_mask[:, n + k:] = 0  # Zero out padding portion in the attention mask

    print(f"Mask Shape: {mask.shape}")
    print(f"Padding Mask Shape: {padding_mask.shape}")

    return mask, padding_mask

# Training class with both AR and Seq2Seq loss
class CombinedTrainer:
    def __init__(self, cfg, train_loader, val_loader, tokenizer):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        total_steps = len(train_loader) * cfg["num_epochs"]
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.num_epochs = cfg["num_epochs"]
        self.total_steps = total_steps
        self.results = []
        self.tokenizer = tokenizer

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            for i, batch in enumerate(self.train_loader):
                # Extract CURRENT_CONV as AR part and PREVIOUS_CONV as Seq2Seq part
                current_conv = batch['CURRENT_CONV'].to(self.device)
                previous_conv = batch['PREVIOUS_CONV'].to(self.device)
                input_ids = batch['PADDED_INPUT'].to(self.device)

                n = current_conv.size(0)
                k = previous_conv.size(0)

                # Ensure the total length is calculated correctly for mask
                total_len = n + k

                # Create a composite attention mask
                mask, padding_mask = prepare_masks(n, k, total_len, self.device)

                # Print one example of data being fed into the model (input IDs and attention mask)
                if i == 0:
                    print("Example Input IDs (first 10 tokens):", input_ids[0, :10])
                    print("Example Attention Mask (first 10 tokens):", mask[0, :10])
                    print("Full Input IDs Shape:", input_ids.shape)
                    print("Full Attention Mask Shape:", mask.shape)
                    print(f"Boundary between AR and Seq2Seq: n={n}, k={k}")

                # Forward pass through the model
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=input_ids)

                # Split outputs
                output_ar = outputs.logits[:, :n, :]  # First n tokens for AR
                output_seq2seq = outputs.logits[:, n:n+k, :]  # Last k tokens for Seq2Seq

                # Print outputs for inspection
                if i == 0:
                    print("Output (AR Part):", output_ar[0, :5])  # First 5 tokens of AR part output
                    print("Output (Seq2Seq Part):", output_seq2seq[0, :5])  # First 5 tokens of Seq2Seq part output

                # Compute losses
                loss_ar = F.cross_entropy(output_ar.view(-1, self.model.config.vocab_size), current_conv.view(-1))
                loss_seq2seq = F.cross_entropy(output_seq2seq.view(-1, self.model.config.vocab_size), previous_conv.view(-1))

                # Print losses for validation
                print(f"Step {i+1}/{len(self.train_loader)} | AR Loss: {loss_ar.item()}, Seq2Seq Loss: {loss_seq2seq.item()}")

                # Combine losses (with equal weighting, adjust as necessary)
                total_loss = (loss_ar + loss_seq2seq) / 2

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Step {i+1}/{len(self.train_loader)}: Total Loss = {total_loss.item()} | Total Steps: {self.total_steps}")
                self.results.append({
                    'epoch': epoch + 1,
                    'train_loss': total_loss.item(),
                })

                # Early stopping for validation
                if i + 1 == self.cfg["early_stop_step"]:
                    print("Early stopping triggered. Running validation...")
                    self.validate()
                    return

            self.validate()

        print("Training completed.")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract CURRENT_CONV as AR part and PREVIOUS_CONV as Seq2Seq part
                current_conv = batch['CURRENT_CONV'].to(self.device)
                previous_conv = batch['PREVIOUS_CONV'].to(self.device)
                input_ids = batch['PADDED_INPUT'].to(self.device)

                n = current_conv.size(0)
                k = previous_conv.size(0)

                # Ensure the total length is calculated correctly for mask
                total_len = n + k

                # Create a composite attention mask
                mask, padding_mask = prepare_masks(n, k, total_len, self.device)

                # Forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=input_ids)

                # Split outputs
                output_ar = outputs.logits[:, :n, :]  # First n tokens for AR
                output_seq2seq = outputs.logits[:, n:n+k, :]  # Last k tokens for Seq2Seq

                # Compute losses
                loss_ar = F.cross_entropy(output_ar.view(-1, self.model.config.vocab_size), current_conv.view(-1))
                loss_seq2seq = F.cross_entropy(output_seq2seq.view(-1, self.model.config.vocab_size), previous_conv.view(-1))

                # Combine losses (with equal weighting, adjust as necessary)
                total_loss = (loss_ar + loss_seq2seq) / 2

                # Accumulate validation loss
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss}")
        self.results.append({
            'epoch': 'validation',
            'val_loss': avg_val_loss,
        })

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(self.cfg["experiment_dir"], "training_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)

    # Load the tokenized dataset
    with open(cfg["sequential_data_path"], 'r') as f:
        data = json.load(f)
    
    # Calculate the split index for validation
    split_index = int(len(data) * 0.95)
    
    # Create train and validation datasets
    train_dataset = SequentialMemoryPileDataset(cfg["sequential_data_path"], tokenizer, max_len=cfg["max_tokens"], start_idx=0, end_idx=split_index)
    val_dataset = SequentialMemoryPileDataset(cfg["sequential_data_path"], tokenizer, max_len=cfg["max_tokens"], start_idx=split_index)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

    trainer = CombinedTrainer(cfg, train_loader, val_loader, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
