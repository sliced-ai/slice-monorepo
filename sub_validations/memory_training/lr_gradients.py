import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random

# Updated Configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "batch_size": 2,  # Batch size of 2: AR and Seq2Seq for a single data point
    "gpu_device": "cuda:1",
    "sequential_data_path": "/workspace/slice-monorepo/sub_validations/memory_training/output_large.json",
    "max_tokens": 2000,  # Max tokens per batch item set to 2000
    "num_epochs": 1,  # Set to 1 for faster experimentation
    "experiment_dir": "/workspace/slice-monorepo/experiments",  # Base directory for experiments
    "experiment_name": "experiment_4",  # Name of the experiment
    "early_stop_step": None,  # No early stop for validation
    "learning_rates": [1e-3, 1e-4,1e-5, 1e-6, 1e-7],  # List of learning rates
    "percentages": [0.05, 0.2,0.1, 0.1, 0.45]  # Corresponding percentages of parameters for each learning rate
}

# Dataset class for the AR and Seq2Seq data
class SequentialMemoryPileDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the AR and Seq2Seq data from the dataset
        ar_data = self.data[idx]['ar']
        seq2seq_data = self.data[idx]['seq2seq']

        # Tokenize the AR and Seq2Seq data
        ar_tokens = self.tokenizer(ar_data, truncation=True, padding=False, max_length=self.max_len, return_tensors='pt')
        seq2seq_tokens = self.tokenizer(seq2seq_data, truncation=True, padding=False, max_length=self.max_len, return_tensors='pt')

        # Ensure the input_ids are in LongTensor format
        ar_input_ids = ar_tokens['input_ids'].squeeze().long()
        seq2seq_input_ids = seq2seq_tokens['input_ids'].squeeze().long()

        # Create attention masks
        ar_attention_mask = torch.ones_like(ar_input_ids)
        seq2seq_attention_mask = torch.ones_like(seq2seq_input_ids)

        return {
            'ar_input_ids': ar_input_ids,
            'seq2seq_input_ids': seq2seq_input_ids,
            'ar_attention_mask': ar_attention_mask,
            'seq2seq_attention_mask': seq2seq_attention_mask,
            'ar_labels': ar_input_ids.clone(),
            'seq2seq_labels': seq2seq_input_ids.clone()
        }

# Training class with custom learning rates
class CombinedTrainer:
    def __init__(self, cfg, train_loader, val_loader, tokenizer):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.results = []
        self.tokenizer = tokenizer

        # Create the experiment directory
        self.experiment_dir = os.path.join(cfg["experiment_dir"], cfg["experiment_name"])
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Assign learning rates to parameters
        self.param_groups = self.assign_lr_to_params(self.model, cfg["learning_rates"], cfg["percentages"])
        self.optimizer = optim.AdamW(self.param_groups)
        self.num_epochs = cfg["num_epochs"]

    def assign_lr_to_params(self, model, learning_rates, percentages):
        # Get all parameters
        all_params = list(model.named_parameters())

        # Shuffle parameters randomly
        random.shuffle(all_params)

        # Create parameter groups based on the percentages
        param_groups = []
        start_idx = 0

        for lr, percentage in zip(learning_rates, percentages):
            num_params = int(len(all_params) * percentage)
            param_group = [param for name, param in all_params[start_idx:start_idx + num_params]]
            param_groups.append({"params": param_group, "lr": lr})
            start_idx += num_params

        return param_groups

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            for i, batch in enumerate(self.train_loader):
                ar_input_ids = batch['ar_input_ids'].to(self.device)
                seq2seq_input_ids = batch['seq2seq_input_ids'].to(self.device)
                ar_attention_mask = batch['ar_attention_mask'].to(self.device)
                seq2seq_attention_mask = batch['seq2seq_attention_mask'].to(self.device)
                ar_labels = batch['ar_labels'].to(self.device)
                seq2seq_labels = batch['seq2seq_labels'].to(self.device)

                # Forward pass for AR
                self.optimizer.zero_grad()
                ar_outputs = self.model(input_ids=ar_input_ids, attention_mask=ar_attention_mask, labels=ar_labels)
                ar_loss = ar_outputs.loss

                # Forward pass for Seq2Seq
                seq2seq_outputs = self.model(input_ids=seq2seq_input_ids, attention_mask=seq2seq_attention_mask, labels=seq2seq_labels)
                seq2seq_loss = seq2seq_outputs.loss

                # Combine losses (with equal weighting, adjust as necessary)
                total_loss = (ar_loss/2 + seq2seq_loss/1.5)

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()

                # Print losses
                print(f"Step {i+1}/{len(self.train_loader)} | AR Loss: {ar_loss.item()}, Seq2Seq Loss: {seq2seq_loss.item()}, Total Loss: {total_loss.item()}")

                self.results.append({
                    'epoch': epoch + 1,
                    'train_loss': total_loss.item(),
                    'ar_loss': ar_loss.item(),
                    'seq2seq_loss': seq2seq_loss.item(),
                    'step': i + 1
                })

                # Early stopping for validation
                if self.cfg["early_stop_step"] is not None and i + 1 == self.cfg["early_stop_step"]:
                    print("Early stopping triggered. Running validation...")
                    self.validate()
                    return

            self.validate()

        print("Training completed.")
        self.save_model()
        self.save_results()

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_ar_loss = 0
        val_seq2seq_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                ar_input_ids = batch['ar_input_ids'].to(self.device)
                seq2seq_input_ids = batch['seq2seq_input_ids'].to(self.device)
                ar_attention_mask = batch['ar_attention_mask'].to(self.device)
                seq2seq_attention_mask = batch['seq2seq_attention_mask'].to(self.device)
                ar_labels = batch['ar_labels'].to(self.device)
                seq2seq_labels = batch['seq2seq_labels'].to(self.device)

                # Forward pass for AR
                ar_outputs = self.model(input_ids=ar_input_ids, attention_mask=ar_attention_mask, labels=ar_labels)
                ar_loss = ar_outputs.loss

                # Forward pass for Seq2Seq
                seq2seq_outputs = self.model(input_ids=seq2seq_input_ids, attention_mask=seq2seq_attention_mask, labels=seq2seq_labels)
                seq2seq_loss = seq2seq_outputs.loss

                # Combine losses (with equal weighting, adjust as necessary)
                total_loss = (ar_loss + seq2seq_loss) / 2

                # Print validation losses per batch
                print(f"Validation Step {i+1} | AR Loss: {ar_loss.item()}, Seq2Seq Loss: {seq2seq_loss.item()}, Total Loss: {total_loss.item()}")

                # Accumulate validation loss
                val_loss += total_loss.item()
                val_ar_loss += ar_loss.item()
                val_seq2seq_loss += seq2seq_loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_ar_loss = val_ar_loss / len(self.val_loader)
        avg_val_seq2seq_loss = val_seq2seq_loss / len(self.val_loader)

        print(f"Validation Loss: {avg_val_loss} | Avg AR Loss: {avg_val_ar_loss} | Avg Seq2Seq Loss: {avg_val_seq2seq_loss}")
        self.results.append({
            'epoch': 'validation',
            'val_loss': avg_val_loss,
            'val_ar_loss': avg_val_ar_loss,
            'val_seq2seq_loss': avg_val_seq2seq_loss,
        })

    def save_model(self):
        model_save_path = os.path.join(self.experiment_dir, "model_checkpoint.pt")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(self.experiment_dir, "training_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

def main():
    # Use a standard tokenizer (GPT-2 tokenizer as an example)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid issues

    # Load the sequential AR and Seq2Seq data
    with open(cfg["sequential_data_path"], 'r') as f:
        data = json.load(f)
    
    # Calculate the split index for validation (last 10% of the dataset)
    split_index = int(len(data) * 0.9)
    
    # Create train and validation datasets
    train_dataset = SequentialMemoryPileDataset(data[:split_index], tokenizer, max_len=cfg["max_tokens"])
    val_dataset = SequentialMemoryPileDataset(data[split_index:], tokenizer, max_len=cfg["max_tokens"])
    
    # Create data loaders (do not shuffle the data)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    trainer = CombinedTrainer(cfg, train_loader, val_loader, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
