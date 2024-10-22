import os
import json
import torch
import numpy as np
import pandas as pd
import copy
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gc
from glob import glob

# Global Configuration
CONFIG = {
    "gpu": 1  # Specify the GPU to use (0 for the first GPU, 1 for the second GPU, etc.)
}

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def save_config(config, experiment_name):
    config_path = os.path.join('experiments', experiment_name, 'config.json')
    with open(config_path, 'w') as cfg_file:
        json.dump(config, cfg_file, indent=4)

def load_json_files(data_dir, start_idx, end_idx):
    file_paths = sorted(glob(os.path.join(data_dir, '*.json')))
    data = []
    for file_path in file_paths[start_idx:end_idx]:
        with open(file_path, 'r') as data_file:
            data.extend(json.load(data_file))
    return data

def ensure_dirs(cfg):
    exp_dir = os.path.join('experiments', cfg["experiment_name"])
    cfg["main_model"]["save_dir"] = os.path.join(exp_dir, cfg["main_model"]["save_dir"])
    cfg["main_model"]["csv_file_path"] = os.path.join(exp_dir, cfg["main_model"]["csv_file_path"])
    os.makedirs(exp_dir, exist_ok=True)

def gen_lrs(cfg):
    lr_ranges = cfg["learning_rate_ranges"]
    lrs = []
    for lr_range in lr_ranges:
        start, end, samples = lr_range["start"], lr_range["end"], lr_range["samples"]
        lrs.extend(np.linspace(start, end, samples))
    return lrs

def save_res(results, csv_path):
    df = pd.DataFrame(results)
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def clean_mem(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

class UtteranceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name_utterance = self.data[idx]
        text = f"{name_utterance['name']}: {name_utterance['utterance']}"
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }

class Trainer:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.utterance_ds = UtteranceDataset(data, self.tokenizer, max_len=cfg["max_length"])
        self.dl = DataLoader(self.utterance_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)
        self.single_batch = next(iter(self.dl))
        self.batch_size = cfg["batch_size"]
        self.device = torch.device(f'cuda:{CONFIG["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.original_model = self.load_model(cfg["main_model"]["name"]).eval()
        self.original_model_state_dict = {k: v.clone().detach().to(self.device) for k, v in self.original_model.state_dict().items()}

    def load_model(self, original_model_name):
        return GPTNeoXForCausalLM.from_pretrained(original_model_name).to(self.device)

    def copy_model(self):
        # Deep copy the original model to create a new instance on the GPU
        model_copy = copy.deepcopy(self.original_model)
        model_copy.load_state_dict(self.original_model_state_dict, strict=False)
        return model_copy

    def train_with_fixed_lr(self, lr, start_idx, end_idx):
        # Copy the original model in GPU memory to initialize the training model
        model = self.copy_model()
        opt = optim.AdamW(model.parameters(), lr=lr)
        results = []

        epoch = 0
        while epoch < self.cfg["num_epochs"]:
            for batch_start in range(start_idx, end_idx, self.cfg["batch_size"]):
                batch_end = min(batch_start + self.cfg["batch_size"], end_idx)
                if batch_start >= len(self.data):
                    break
                current_batch_data = self.data[batch_start:batch_end]
                current_batch = UtteranceDataset(current_batch_data, self.tokenizer, self.cfg["max_length"])
                current_dl = DataLoader(current_batch, batch_size=self.cfg["batch_size"], shuffle=False, pin_memory=True)
                batch = next(iter(current_dl))
                model.train()
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                opt.zero_grad()
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()
                opt.step()

                avg_train_loss = loss.item()
                overall_mad = self.calculate_mad_gpu(model)

                results.append({
                    "LR": lr,
                    "Epoch": epoch + 1,
                    "Batch Start": batch_start,
                    "Batch End": batch_end,
                    "Train Loss": avg_train_loss,
                    "Overall MAD": overall_mad
                })

                print(f"Epoch {epoch + 1}, Batch {batch_start}-{batch_end}: Train Loss = {avg_train_loss}, Overall MAD = {overall_mad}")

            epoch += 1

        return results

    def calculate_mad_gpu(self, model):
        layer_indices = np.linspace(0, len(list(model.named_parameters())) - 1, int(len(list(model.named_parameters())) * 0.25), dtype=int)
        selected_layers = [list(model.named_parameters())[i] for i in layer_indices]

        mad_values = []
        for name, param in selected_layers:
            original_param = self.original_model_state_dict[name]
            mad = torch.mean(torch.abs(param - original_param)).item()
            mad_values.append(mad)

        return np.mean(mad_values)

    def manual_gc(self):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    cfg = load_config('config.json')
    experiment_name = cfg["experiment_name"]
    start_idx = cfg["start_idx"]
    end_idx = cfg["end_idx"]
    data = load_json_files(cfg['data_dir'], start_idx, end_idx)
    ensure_dirs(cfg)
    save_config(cfg, experiment_name)

    lrs = gen_lrs(cfg)

    # Initialize the Trainer and load the original model once
    trainer = Trainer(cfg, data)

    for lr in lrs:
        results = trainer.train_with_fixed_lr(lr, start_idx, end_idx)
        save_res(results, cfg["main_model"]["csv_file_path"])
        trainer.manual_gc()

if __name__ == "__main__":
    main()
