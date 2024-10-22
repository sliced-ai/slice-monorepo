import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import struct
from torch.utils.data.dataloader import default_collate
import random
# Configuration
config = {
    "experiment_name": "1_epoch_410m_higherlr",
    "index_file_path": "/workspace/data/unsharded/document.idx",
    "bin_file_path": "/workspace/data/unsharded/document.bin",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",  # Tokenizer path
    "main_model": {
        "name": "EleutherAI/pythia-410m",
        "save_dir": "model_saves",
        "gpu_device": "cuda:1"  # Set the device to cuda:1
    },
    "batch_size": 2,
    "max_length": 2049,
    "num_epochs": 1,
    "subset_range": [0, 10000],
    "num_model_saves_per_epoch": 8,
    "learning_rates": [1e-4, 1e-6],  # List of learning rates
    "percentages": [0.2, 0.8]  # Corresponding percentages of parameters for each learning rate
}

def ensure_dirs(cfg):
    exp_dir = os.path.join('experiments', cfg["experiment_name"])
    cfg["main_model"]["save_dir"] = os.path.join(exp_dir, cfg["main_model"]["save_dir"])
    cfg["main_model"]["csv_file_path"] = os.path.join(exp_dir, "batch_training_results.csv")
    os.makedirs(cfg["main_model"]["save_dir"], exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

def save_res(results, csv_path):
    df = pd.DataFrame(results)
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

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

        # Check for invalid data (empty or incorrect size)
        if len(tokens) == 0 or len(tokens) > self.max_len:
            return None  # Skip this data entry

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'index': idx
        }

class Trainer:
    def __init__(self, cfg, tokenizer, dataset):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = cfg["main_model"]["gpu_device"]  # Use the configured GPU device
        self.dl = DataLoader(self.dataset, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.results = []

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

    def train_with_custom_lr(self, learning_rates, percentages, num_epochs):
        # Assign learning rates to parameters
        param_groups = self.assign_lr_to_params(self.model, learning_rates, percentages)

        # Create optimizer with parameter groups
        opt = optim.AdamW(param_groups)

        num_batches = len(self.dl)
        save_intervals = num_batches // self.cfg.get("num_model_saves_per_epoch", 4)
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            for i, batch in enumerate(self.dl):
                if batch is None:  # Skip batches with None
                    continue
                # Check if the batch has valid data (non-empty tensors)
                if any(tensor.size(0) == 0 for tensor in batch['input_ids']):
                    continue  # Skip this batch if it contains invalid data
                
                batch_inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != 'index'}
                opt.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                opt.step()
                
                avg_train_loss = loss.item()

                print(f"Epoch {epoch}, Batch {i + 1}: Train Loss = {avg_train_loss}")

                self.results.append({
                    "epoch": epoch,
                    "batch": i + 1,
                    "data_indices": batch['index'].tolist(),
                    "train_loss": avg_train_loss
                })
                self.save_results()

        final_model_path = os.path.join(self.cfg["main_model"]["save_dir"], "final_model.pt")
        torch.save(self.model.state_dict(), final_model_path)
        return final_model_path

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.cfg["main_model"]["csv_file_path"], index=False)

def main():
    cfg = config  # Use the embedded configuration
    ensure_dirs(cfg)

    dtype, sizes, pointers, doc_idx = load_index_file(cfg['index_file_path'])

    # Load the tokenizer from a local JSON file path
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"])

    dataset = PileDataset(cfg['bin_file_path'], pointers, sizes, dtype, tokenizer, max_len=cfg["max_length"])
    trainer = Trainer(cfg, tokenizer, dataset)

    learning_rates = cfg["learning_rates"]  # List of learning rates
    percentages = cfg["percentages"]  # Corresponding percentages

    num_epochs = cfg["num_epochs"]
    final_model_path = trainer.train_with_custom_lr(learning_rates, percentages, num_epochs)
    print(f"Training completed. Final model saved at: {final_model_path}")

if __name__ == "__main__":
    main()
