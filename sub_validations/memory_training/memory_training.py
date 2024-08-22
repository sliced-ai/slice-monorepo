import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import struct
from torch.utils.data._utils.collate import default_collate

# Define the configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "experiment_name": "combined_training_7",
    "starting_learning_rate": 1e-5,
    "batch_size": 1,
    "gpu_device": "cuda:1",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "experiments_dir": "/workspace/slice-monorepo/sub_validations/memory_training/experiments",
    "num_epochs": 10,
    "pile_data": {
        "index_file_path": "/workspace/data/unsharded/document.idx",
        "bin_file_path": "/workspace/data/unsharded/document.bin",
        "max_size": 2049,
    },
    "sequential_data_path": "/workspace/slice-monorepo/sub_validations/memory_training/tokenized_output_10k.json",
    "max_steps_per_epoch": None,
    "max_tokens": 4098  # Maximum tokens in a batch
}

class SequentialMemoryPileDataset(Dataset):
    def __init__(self, token_file, max_len):
        with open(token_file, 'r') as f:
            self.tokens = json.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_sequence = self.tokens[idx]
        if len(token_sequence) > self.max_len:
            token_sequence = token_sequence[:self.max_len]
        return {
            'input_ids': torch.tensor(token_sequence, dtype=torch.long),
            'index': idx
        }

class PileDataset(Dataset):
    def __init__(self, bin_path, pointers, sizes, dtype, tokenizer, max_len):
        self.bin_path = bin_path
        self.pointers = pointers
        self.sizes = sizes
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pointers)

    def __getitem__(self, idx):
        with open(self.bin_path, "rb") as f:
            f.seek(self.pointers[idx])
            entry = f.read(self.sizes[idx] * self.dtype().itemsize)
        
        tokens = np.frombuffer(entry, dtype=self.dtype).tolist()
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'index': idx
        }

def load_index_file(index_path):
    with open(index_path, "rb") as f:
        magic_test = f.read(9)
        assert magic_test == b"MMIDIDX\x00\x00", "Index file doesn't match expected format."

        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, "Unsupported index file version."

        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float32,
            7: np.float64,
            8: np.uint16,
        }[dtype_code]

        length = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]

        sizes = np.frombuffer(f.read(length * 4), dtype=np.int32)
        pointers = np.frombuffer(f.read(length * 8), dtype=np.int64)
        doc_idx = np.frombuffer(f.read(doc_count * 8), dtype=np.int64)

    return dtype, sizes, pointers, doc_idx

def custom_collate_fn(batch):
    # Filter out any None items from the batch
    batch = [item for item in batch if item is not None]
    return default_collate(batch) if batch else None

class CombinedTrainer:
    def __init__(self, cfg, seq_loader, pile_loader, tokenizer):
        self.cfg = cfg
        self.seq_loader = seq_loader
        self.pile_loader = pile_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        total_steps = len(seq_loader) * cfg["num_epochs"]
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.num_epochs = cfg["num_epochs"]
        self.results = []
        self.tokenizer = tokenizer
        self.max_tokens = cfg["max_tokens"]
        self.first_batch_logged = False  # To ensure only the first batch is logged

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            seq_iter = iter(self.seq_loader)
            pile_iter = iter(self.pile_loader)

            epoch_loss = 0
            steps = 0

            while True:
                if self.cfg["max_steps_per_epoch"] is not None and steps >= self.cfg["max_steps_per_epoch"]:
                    break

                # Sequential Data
                seq_batch = next(seq_iter, None)
                if seq_batch is None or not isinstance(seq_batch, dict):
                    print(f"Invalid or insufficient sequential data at step {steps + 1}. Ending epoch early.")
                    break

                seq_tokens = seq_batch['input_ids'].flatten()  # Flatten to ensure a single sequence
                remaining_tokens = self.max_tokens - seq_tokens.size(0)

                pile_tokens = []
                pile_sizes = []  # Store the size of each Pile data item

                # Fill remaining tokens with Pile data
                while remaining_tokens > 0:
                    pile_batch = next(pile_iter, None)
                    while pile_batch is None or not isinstance(pile_batch, dict):
                        pile_batch = next(pile_iter, None)
                        if pile_batch is None:
                            pile_iter = iter(self.pile_loader)
                            pile_batch = next(pile_iter, None)

                    pile_item_tokens = pile_batch['input_ids'].flatten().tolist()
                    pile_sizes.append(len(pile_item_tokens))  # Track the size of each Pile data item

                    pile_tokens.extend(pile_item_tokens)

                    if len(pile_tokens) >= remaining_tokens:
                        pile_tokens = pile_tokens[:remaining_tokens]
                        break
                    remaining_tokens -= len(pile_tokens)

                pile_tokens = torch.tensor(pile_tokens, dtype=torch.long).flatten()  # Flatten Pile tokens

                total_batch_size = pile_tokens.size(0) + seq_tokens.size(0)

                # Ensure total batch size is always max_tokens (4098)
                if total_batch_size < self.max_tokens:
                    # Padding if necessary (this case should rarely happen due to filling with Pile data)
                    padding_size = self.max_tokens - total_batch_size
                    padded_tokens = torch.cat([seq_tokens, pile_tokens, torch.zeros(padding_size, dtype=torch.long)], dim=0)
                else:
                    padded_tokens = torch.cat([seq_tokens, pile_tokens], dim=0)[:self.max_tokens]

                if not self.first_batch_logged:
                    # Print out the size of pile and sequential parts for the first batch only
                    print(f"First Batch Debug Info:")
                    print(f"Sequential Tokens Size: {seq_tokens.size(0)}")
                    print(f"Pile Tokens Sizes: {pile_sizes}")
                    print(f"Pile Tokens Total Size: {pile_tokens.size(0)}")
                    print(f"Total Batch Size (should equal 4098): {padded_tokens.size(0)}")
                    self.first_batch_logged = True

                padded_tokens = padded_tokens.unsqueeze(0)  # Add batch dimension

                batch_inputs = {'input_ids': padded_tokens.to(self.device)}

                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

                # Print loss for every batch
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Step {steps + 1}: Train Loss = {loss.item()}, Learning Rate = {self.scheduler.get_last_lr()[0]}")

                self.results.append({
                    'epoch': epoch + 1,
                    'step': steps + 1,
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

                steps += 1

            final_model_path = os.path.join(self.cfg["experiment_dir"], f"{self.cfg['experiment_name']}_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), final_model_path)
            self.save_results()

        print(f"Training completed. Final model saved at: {final_model_path}")

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(self.cfg["experiment_dir"], "training_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

def save_config(cfg, experiment_dir):
    config_path = os.path.join(experiment_dir, 'combined_config.json')
    with open(config_path, 'w') as cfg_file, open('combined_config.json', 'w') as local_cfg_file:
        json.dump(cfg, cfg_file, indent=4)
        json.dump(cfg, local_cfg_file, indent=4)
    return config_path

def ensure_dirs(cfg):
    exp_dir = os.path.join(cfg["experiments_dir"], cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def main():
    experiment_dir = ensure_dirs(cfg)
    cfg["experiment_dir"] = experiment_dir
    save_config(cfg, experiment_dir)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)

    # Create datasets and data loaders
    seq_dataset = SequentialMemoryPileDataset(cfg["sequential_data_path"], max_len=cfg["max_tokens"])
    seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)

    dtype, sizes, pointers, doc_idx = load_index_file(cfg['pile_data']['index_file_path'])
    pile_dataset = PileDataset(cfg['pile_data']['bin_file_path'], pointers, sizes, dtype, tokenizer, max_len=cfg['pile_data']['max_size'])
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)

    # Print out the number of steps and epochs
    total_steps = len(seq_loader) * cfg["num_epochs"]
    cfg["total_steps"] = total_steps
    print(f"Total Steps: {total_steps}, Number of Epochs: {cfg['num_epochs']}")

    # Pass the initialized loaders to the trainer
    trainer = CombinedTrainer(cfg, seq_loader, pile_loader, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
