import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gc

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
        if end_idx > len(self.tokens):
            raise IndexError
        token_sequence = self.tokens[start_idx:end_idx]
        return {
            'input_ids': torch.tensor(token_sequence, dtype=torch.long)
        }

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def clean_mem(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

def retrain_model(cfg, start_window, end_window):
    device = torch.device(cfg["gpu_device"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg['tokenizer_path'])
    
    dataset = RollingWindowDataset('tokenized_utterances.pt', window_size=cfg["window_size"], step_size=cfg["step_size"])
    
    # Ensure the selected range is within valid bounds
    if end_window > len(dataset):
        end_window = len(dataset)
    if start_window >= end_window:
        raise ValueError("Invalid training range specified.")

    # Select the specified window range from the dataset
    selected_indices = list(range(start_window, end_window))
    selected_dataset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = DataLoader(selected_dataset, batch_size=1, shuffle=False, pin_memory=True)

    experiment_dir = os.path.join('experiments', cfg['experiment_name'])
    retrain_dir = os.path.join(experiment_dir, f"retrain_{start_window}_{end_window}")
    os.makedirs(retrain_dir, exist_ok=True)

    retrained_model_path = os.path.join(retrain_dir, "retrained_model.pt")
    if os.path.exists(retrained_model_path):
        print(f"Retrained model already exists at {retrained_model_path}. Skipping retraining.")
        return retrained_model_path

    final_model_path = os.path.join(experiment_dir, cfg["main_model"]["save_dir"], "final_model.pt")
    model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["fixed_learning_rate"])

    model.train()
    for i, batch in enumerate(dataloader):
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        opt.zero_grad()
        outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        opt.step()
        print(f"Processed Window {i + 1}/{len(dataloader)}: Train Loss = {loss.item()}")

    torch.save(model.state_dict(), retrained_model_path)
    clean_mem(model)
    return retrained_model_path

def calculate_inference_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_inference_losses = []

    with torch.no_grad():
        for batch in dataloader:
            batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            all_inference_losses.append(loss.item())

    return total_loss / num_batches, all_inference_losses

def main():
    config_path = 'rolling_window_config.json'
    cfg = load_config(config_path)
    experiment_name = cfg['experiment_name']
    device = cfg['gpu_device']

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg['tokenizer_path'])

    # Define the training range
    start_window = 8000
    end_window = 8500

    # Retrain the model in the specified range
    retrained_model_path = retrain_model(cfg, start_window, end_window)

    # Define the inference range as ±1000 around the max of the training range
    inference_start_window = max(0, end_window - 1000)
    inference_end_window = min(len(torch.load('tokenized_utterances.pt')), end_window + 1000)
    
    # Load the inference dataset
    dataset = RollingWindowDataset('tokenized_utterances.pt', window_size=cfg["window_size"], step_size=cfg["step_size"])
    selected_indices = list(range(inference_start_window, inference_end_window))
    selected_dataset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = DataLoader(selected_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
    model.load_state_dict(torch.load(retrained_model_path, map_location=device))
    model.to(device)

    retrain_dir = os.path.join('experiments', experiment_name, f"retrain_{start_window}_{end_window}")
    inference_results_path = os.path.join(retrain_dir, "retrained_inference_results.csv")
    if os.path.exists(inference_results_path):
        print(f"Inference results already exist at {inference_results_path}. Skipping inference calculation.")
    else:
        _, all_inference_losses = calculate_inference_loss(model, dataloader, device)
        new_inference_results = [{'window': i + 1, 'inference_loss': loss} for i, loss in enumerate(all_inference_losses)]
        pd.DataFrame(new_inference_results).to_csv(inference_results_path, index=False)
        print(f"New inference results saved to {inference_results_path}")

if __name__ == "__main__":
    main()
