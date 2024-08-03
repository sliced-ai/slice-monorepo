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
import struct
from glob import glob

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def save_config(cfg, save_path):
    with open(save_path, 'w') as cfg_file:
        json.dump(cfg, cfg_file, indent=4)

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

def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()

class PileDataset(Dataset):
    def __init__(self, bin_path, pointers, sizes, dtype, tokenizer, max_len, subset_range):
        self.bin_path = bin_path
        self.pointers = pointers
        self.sizes = sizes
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subset_range = subset_range
        self.filtered_indices = [i for i in range(len(pointers)) if subset_range[0] <= i < subset_range[1]]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        with open(self.bin_path, "rb") as f:
            f.seek(self.pointers[actual_idx])
            entry = f.read(self.sizes[actual_idx] * self.dtype().itemsize)
        
        tokens = np.frombuffer(entry, dtype=self.dtype).tolist()
        text = self.tokenizer.decode(tokens)
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'index': actual_idx
        }

class Trainer:
    def __init__(self, cfg, tokenizer, dataset):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dl = DataLoader(self.dataset, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
        self.results = []
        self.original_model = self.load_model(cfg["main_model"]["name"]).eval()
        self.original_model_state_dict = {k: v.clone().detach().to('cuda') for k, v in self.original_model.state_dict().items()}

    def load_model(self, original_model_name):
        return GPTNeoXForCausalLM.from_pretrained(original_model_name).to('cuda')

    def copy_model(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.load_state_dict(self.original_model_state_dict, strict=False)
        return model_copy

    def train_with_fixed_lr(self, lr, num_epochs, calc_inference_loss):
        model = self.copy_model()
        opt = optim.AdamW(model.parameters(), lr=lr)
        num_batches = len(self.dl)
        save_intervals = num_batches // self.cfg.get("num_model_saves_per_epoch", 4)
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            for i, batch in enumerate(self.dl):
                batch_inputs = {k: v.to('cuda', non_blocking=True) for k, v in batch.items() if k != 'index'}
                opt.zero_grad()
                outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                opt.step()
                
                avg_train_loss = loss.item()

                inference_loss = 0
                if calc_inference_loss:
                    clean_mem()
                    with torch.no_grad():
                        inference_outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
                        inference_loss = inference_outputs.loss.item()
                    clean_mem()

                overall_mad = self.calculate_mad_gpu(model)
                
                print(f"Epoch {epoch}, Batch {i + 1}: Train Loss = {avg_train_loss}, Inference Loss = {inference_loss}, Overall MAD = {overall_mad}")

                self.results.append({
                    "epoch": epoch,
                    "batch": i + 1,
                    "data_indices": batch['index'].tolist(),
                    "train_loss": avg_train_loss,
                    "inference_loss": inference_loss,
                    "overall_mad": overall_mad
                })
                self.save_results()

                # Save intermediate models
                if (i + 1) % save_intervals == 0:
                    intermediate_model_path = os.path.join(self.cfg["main_model"]["save_dir"], f"model_epoch_{epoch}_batch_{i + 1}.pt")
                    torch.save(model.state_dict(), intermediate_model_path)

            final_epoch_model_path = os.path.join(self.cfg["main_model"]["save_dir"], f"model_epoch_{epoch}_end.pt")
            torch.save(model.state_dict(), final_epoch_model_path)

        final_model_path = os.path.join(self.cfg["main_model"]["save_dir"], "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        return final_model_path

    def calculate_mad_gpu(self, model):
        layer_indices = np.linspace(0, len(list(model.named_parameters())) - 1, int(len(list(model.named_parameters())) * 0.25), dtype=int)
        selected_layers = [list(model.named_parameters())[i] for i in layer_indices]

        mad_values = []
        for name, param in selected_layers:
            original_param = self.original_model_state_dict[name]
            mad = torch.mean(torch.abs(param - original_param)).item()
            mad_values.append(mad)

        return np.mean(mad_values)

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.cfg["main_model"]["csv_file_path"], index=False)

    def manual_gc(self):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    cfg = load_config('config.json')
    index_file_path = cfg['index_file_path']
    bin_file_path = cfg['bin_file_path']
    experiment_name = cfg['experiment_name']

    ensure_dirs(cfg)

    # Save a copy of the config file to the experiment directory
    config_save_path = os.path.join('experiments', cfg["experiment_name"], 'config.json')
    save_config(cfg, config_save_path)

    dtype, sizes, pointers, doc_idx = load_index_file(index_file_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PileDataset(bin_file_path, pointers, sizes, dtype, tokenizer, max_len=cfg["max_length"], subset_range=cfg["subset_range"])
    trainer = Trainer(cfg, tokenizer, dataset)

    fixed_lr = cfg["fixed_learning_rate"]
    num_epochs = cfg["num_epochs"]
    calc_inference_loss = cfg.get("calc_inference_loss", True)  # Default to True if not specified
    final_model_path = trainer.train_with_fixed_lr(fixed_lr, num_epochs, calc_inference_loss)
    print(f"Training completed. Final model saved at: {final_model_path}")

if __name__ == "__main__":
    main()
