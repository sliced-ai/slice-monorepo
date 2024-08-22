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
    "batch_size": 2,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "experiments_dir": "/workspace/slice-monorepo/sub_validations/cl_scaling/pile_dnd/experiments",
    "num_epochs": 10,
    "pile_data": {
        "index_file_path": "/workspace/data/unsharded/document.idx",
        "bin_file_path": "/workspace/data/unsharded/document.bin",
        "max_size": 2049,
    },
    "window_data_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/dnd/tokenized_utterances.pt",
    "step_size_percentage": 25,  # Step size percentage of the window size
    "percentage_window": 50,  # Rolling window data percentage
    "max_steps_per_epoch": None,  # Limit the number of training steps per epoch (set to None to disable)
    "max_tokens": 4098  # Maximum tokens in a batch
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

class CombinedTrainer:
    def __init__(self, cfg, rw_loader, pile_loader, tokenizer):
        self.cfg = cfg
        self.rw_loader = rw_loader
        self.pile_loader = pile_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        total_steps = len(rw_loader) * cfg["num_epochs"]
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.num_epochs = cfg["num_epochs"]
        self.results = []
        self.tokenizer = tokenizer

        # Calculate the window size and pile size based on percentage
        self.max_tokens = cfg["max_tokens"]
        self.rw_token_size = int(self.max_tokens * cfg["percentage_window"] / 100)
        self.pile_token_size = self.max_tokens - self.rw_token_size

        # Calculate step size after determining the rolling window size
        self.step_size = int(self.rw_token_size * cfg["step_size_percentage"] / 100)

        # Print the calculated token sizes and step size
        print(f"Max Tokens: {self.max_tokens}, Rolling Window Tokens: {self.rw_token_size}, Pile Tokens: {self.pile_token_size}, Step Size: {self.step_size}")

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            rw_iter = iter(self.rw_loader)
            pile_iter = iter(self.pile_loader)

            epoch_loss = 0
            steps = 0

            while True:
                if self.cfg["max_steps_per_epoch"] is not None and steps >= self.cfg["max_steps_per_epoch"]:
                    break

                # Rolling Window Data
                rw_batch = next(rw_iter, None)
                if rw_batch is None or not isinstance(rw_batch, dict) or rw_batch['input_ids'].shape[1] < self.rw_token_size:
                    print(f"Invalid or insufficient rolling window data at step {steps + 1}. Ending epoch early.")
                    break

                # Pile Data
                pile_tokens = []
                while len(pile_tokens) < self.pile_token_size:
                    pile_batch = next(pile_iter, None)
                    while pile_batch is None or not isinstance(pile_batch, dict):
                        pile_batch = next(pile_iter, None)
                        if pile_batch is None:
                            pile_iter = iter(self.pile_loader)
                            pile_batch = next(pile_iter, None)

                    pile_tokens.extend(pile_batch['input_ids'][0].tolist())

                # Trim excess tokens if necessary
                if len(pile_tokens) > self.pile_token_size:
                    pile_tokens = pile_tokens[:self.pile_token_size]

                pile_tokens = torch.tensor(pile_tokens, dtype=torch.long).unsqueeze(0)

                # Combine Data
                combined_input_ids = torch.cat([rw_batch['input_ids'][:, :self.rw_token_size], pile_tokens], dim=1)
                batch_inputs = {'input_ids': combined_input_ids.to(self.device, non_blocking=True)}

                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

                rw_inference_loss, pile_inference_loss = self.calculate_inference_loss(rw_batch, pile_tokens)

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Step {steps + 1}: Train Loss = {loss.item()}, Learning Rate = {self.scheduler.get_last_lr()[0]}, RW Inference Loss = {rw_inference_loss}, Pile Inference Loss = {pile_inference_loss}")
                
                self.results.append({
                    'epoch': epoch + 1,
                    'step': steps + 1,
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'rw_inference_loss': rw_inference_loss,
                    'pile_inference_loss': pile_inference_loss
                })

                steps += 1

            final_model_path = os.path.join(self.cfg["experiment_dir"], f"{self.cfg['experiment_name']}_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), final_model_path)
            self.save_results()
        print(f"Training completed. Final model saved at: {final_model_path}")

    def calculate_inference_loss(self, rw_batch, pile_tokens):
        self.model.eval()
        rw_inference_loss = 0
        pile_inference_loss = 0

        with torch.no_grad():
            # Calculate inference loss for rolling window data
            rw_batch_inputs = {k: v.to(self.device, non_blocking=True) for k, v in rw_batch.items() if k != 'index'}
            rw_outputs = self.model(**rw_batch_inputs, labels=rw_batch_inputs['input_ids'])
            rw_inference_loss = rw_outputs.loss.item()

            # Calculate inference loss for pile data
            pile_batch_inputs = {'input_ids': pile_tokens.to(self.device, non_blocking=True)}
            pile_outputs = self.model(**pile_batch_inputs, labels=pile_batch_inputs['input_ids'])
            pile_inference_loss = pile_outputs.loss.item()

        self.model.train()
        return rw_inference_loss, pile_inference_loss

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
    
    # Calculate the sizes based on max_tokens and percentage_window
    max_tokens = cfg["max_tokens"]
    rw_token_size = int(max_tokens * cfg["percentage_window"] / 100)
    pile_token_size = max_tokens - rw_token_size
    step_size = int(rw_token_size * cfg["step_size_percentage"] / 100)

    # Print the calculated token sizes and step size
    print(f"Max Tokens: {max_tokens}, Rolling Window Tokens: {rw_token_size}, Pile Tokens: {pile_token_size}, Step Size: {step_size}")

    # Create datasets and data loaders
    rw_dataset = RollingWindowDataset(cfg["window_data_path"], window_size=rw_token_size, step_size=step_size)
    rw_loader = DataLoader(rw_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)

    dtype, sizes, pointers, doc_idx = load_index_file(cfg['pile_data']['index_file_path'])
    pile_dataset = PileDataset(cfg['pile_data']['bin_file_path'], pointers, sizes, dtype, tokenizer, max_len=cfg['pile_data']['max_size'])
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)

    total_steps = len(rw_loader) * cfg["num_epochs"]
    cfg["total_steps"] = total_steps

    # Print out the number of steps and epochs
    print(f"Total Steps: {total_steps}, Number of Epochs: {cfg['num_epochs']}")

    # Now pass the initialized loaders to the trainer
    trainer = CombinedTrainer(cfg, rw_loader, pile_loader, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
