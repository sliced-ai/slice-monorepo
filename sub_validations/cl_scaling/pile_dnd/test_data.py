import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np
import struct
from transformers import PreTrainedTokenizerFast

# Define the configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "batch_size": 2,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "pile_data": {
        "index_file_path": "/workspace/data/unsharded/document.idx",
        "bin_file_path": "/workspace/data/unsharded/document.bin",
        "max_length": 2049
    },
    "window_data_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/dnd/tokenized_utterances.pt",
    "window_size": 2049,
    "step_size_percentage": 25,  # Step size percentage of the window size
    "num_epochs": 10
}

class RollingWindowDataset(Dataset):
    def __init__(self, token_file, window_size, step_size):
        self.tokens = torch.load(token_file)
        self.window_size = window_size
        self.step_size = step_size
        self.num_samples = (len(self.tokens) - window_size) // step_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        if end_idx > len(self.tokens):  # Ignore final data that can't fit into the window
            return None  # Signal invalid data
        token_sequence = self.tokens[start_idx:end_idx]
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
        if len(tokens) != self.max_len:
            return None  # Return None to signal an invalid batch
        
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

class DataTester:
    def __init__(self, cfg, rw_loader, pile_loader):
        self.cfg = cfg
        self.rw_loader = rw_loader
        self.pile_loader = pile_loader

        # Set the max_token_size dynamically based on batch size and data
        self.cfg['max_token_size'] = self.cfg['window_size'] * self.cfg['batch_size']

    def test_data(self):
        for epoch in range(self.cfg['num_epochs']):
            print(f"Starting Epoch {epoch + 1}/{self.cfg['num_epochs']}")
            rw_iter = iter(self.rw_loader)
            pile_iter = iter(self.pile_loader)
            
            for i in range(len(self.rw_loader)):
                # Rolling Window Data
                rw_batch = next(rw_iter, None)
                if rw_batch is None or not isinstance(rw_batch, dict) or rw_batch['input_ids'].shape[1] < self.cfg['window_size']:
                    print(f"Invalid or insufficient rolling window data at step {i + 1}/{len(self.rw_loader)}. Ending epoch early.")
                    break

                # Pile Data
                pile_batch = next(pile_iter, None)
                while pile_batch is None or not isinstance(pile_batch, dict):  # Continue until valid pile data is found
                    pile_batch = next(pile_iter, None)
                    if pile_batch is None:  # If the pile data ends, restart the iterator
                        pile_iter = iter(self.pile_loader)
                        pile_batch = next(pile_iter, None)

                if pile_batch['input_ids'].shape[1] != self.cfg['pile_data']['max_length']:
                    print(f"Skipping invalid pile data at index {pile_batch['index']} with shape {pile_batch['input_ids'].shape}.")
                    continue

                # Combine Data
                combined_input_ids = torch.cat([rw_batch['input_ids'], pile_batch['input_ids']], dim=1)
                combined_input_ids = combined_input_ids[:, :self.cfg['max_token_size']]  # Ensure the max token size is correct

            print(f"Finished processing {i + 1}/{len(self.rw_loader)} steps for the rolling window in Epoch {epoch + 1}")

def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)
    
    step_size = int(cfg["window_size"] * cfg["step_size_percentage"] / 100)
    rw_dataset = RollingWindowDataset(cfg["window_data_path"], window_size=cfg["window_size"], step_size=step_size)
    rw_loader = DataLoader(rw_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    dtype, sizes, pointers, doc_idx = load_index_file(cfg['pile_data']['index_file_path'])
    pile_dataset = PileDataset(cfg['pile_data']['bin_file_path'], pointers, sizes, dtype, tokenizer, max_len=cfg['pile_data']['max_length'])
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    tester = DataTester(cfg, rw_loader, pile_loader)
    tester.test_data()

if __name__ == "__main__":
    main()
