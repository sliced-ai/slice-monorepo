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

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def load_json_files(data_dir):
    file_paths = sorted(glob(os.path.join(data_dir, '*.json')))
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as data_file:
            data.extend(json.load(data_file))
    return data

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

def clean_mem(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

class QADataset(Dataset):
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
            'attention_mask': tokens['attention_mask'].squeeze(),
            'index': idx
        }

class Trainer:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_ds = QADataset(data, self.tokenizer, max_len=cfg["max_length"])
        self.dl = DataLoader(self.qa_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)
        self.results = []
        self.original_model = self.load_model(cfg["main_model"]["name"]).eval()
        self.original_model_state_dict = {k: v.clone().detach().to('cuda') for k, v in self.original_model.state_dict().items()}

    def load_model(self, original_model_name):
        return GPTNeoXForCausalLM.from_pretrained(original_model_name).to('cuda')

    def copy_model(self):
        # Deep copy the original model to create a new instance on the GPU
        model_copy = copy.deepcopy(self.original_model)
        model_copy.load_state_dict(self.original_model_state_dict, strict=False)
        return model_copy

    def train_with_fixed_lr(self, lr, num_epochs):
        model = self.copy_model()
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            num_batches_per_epoch = len(self.dl)
            save_intervals = num_batches_per_epoch // 4

            for i, batch in enumerate(self.dl):
                batch_inputs = {k: v.to('cuda', non_blocking=True) for k, v in batch.items() if k != 'index'}
                batch_indices = batch['index'].tolist()
                opt.zero_grad()
                outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                opt.step()
                
                avg_train_loss = loss.item()
                overall_mad = self.calculate_mad_gpu(model)
                
                print(f"Epoch {epoch}, Batch {i + 1}: Train Loss = {avg_train_loss}, Overall MAD = {overall_mad}")

                self.results.append({
                    "epoch": epoch,
                    "batch": i + 1,
                    "data_indices": batch_indices,
                    "train_loss": avg_train_loss,
                    "overall_mad": overall_mad
                })
                self.save_results()

                # Save intermediate models
                if (i + 1) % save_intervals == 0:
                    intermediate_model_path = os.path.join(self.cfg["main_model"]["save_dir"], f"model_epoch_{epoch}_batch_{i + 1}.pt")
                    torch.save(model.state_dict(), intermediate_model_path)

            # Save model at the end of each epoch
            final_epoch_model_path = os.path.join(self.cfg["main_model"]["save_dir"], f"model_epoch_{epoch}_end.pt")
            torch.save(model.state_dict(), final_epoch_model_path)

        # Save the final model at the end of training
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
    data_dir = cfg['data_dir']
    experiment_name = cfg['experiment_name']
    
    data = load_json_files(data_dir)
    
    print(f"Loaded {len(data)} utterances from {len(glob(os.path.join(data_dir, '*.json')))} files.")
    print("Sample data:", data[:2])  # Print a sample of the data
    
    ensure_dirs(cfg)
    
    trainer = Trainer(cfg, data)

    # Print out examples of how the data looks for training
    for i in range(2):
        example = trainer.qa_ds[i]
        input_ids = example['input_ids']
        decoded_text = trainer.tokenizer.decode(input_ids)
        attention_mask = example['attention_mask']
        print(f"Example {i + 1} - Decoded Text: {decoded_text}")
        print(f"Example {i + 1} - Attention Mask: {attention_mask.tolist()}")
    
    # Train the model with a fixed learning rate
    fixed_lr = cfg["fixed_learning_rate"]
    num_epochs = cfg["num_epochs"]
    final_model_path = trainer.train_with_fixed_lr(fixed_lr, num_epochs)
    print(f"Training completed. Final model saved at: {final_model_path}")

if __name__ == "__main__":
    main()
