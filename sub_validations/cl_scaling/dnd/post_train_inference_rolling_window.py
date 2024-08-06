import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from glob import glob

class RollingWindowDataset(Dataset):
    def __init__(self, token_file, window_size, step_size):
        # Load the full list of tokens
        self.tokens = torch.load(token_file)
        self.window_size = window_size
        self.step_size = step_size
        # Calculate the number of samples
        self.num_samples = (len(self.tokens) - window_size) // step_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        if end_idx > len(self.tokens):  # Ignore final data that can't fit into the window
            raise IndexError
        token_sequence = self.tokens[start_idx:end_idx]
        return {
            'input_ids': torch.tensor(token_sequence, dtype=torch.long)
        }

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

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
    tokenizer_path = cfg['tokenizer_path']
    device = cfg['gpu_device']

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Load the tokenized data
    dataset = RollingWindowDataset('tokenized_utterances.pt', window_size=cfg["window_size"], step_size=cfg["step_size"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    experiment_dir = os.path.join('experiments', experiment_name)
    final_model_path = os.path.join(experiment_dir, cfg["main_model"]["save_dir"], "final_model.pt")
    model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    model.to(device)

    inference_results_path = os.path.join(experiment_dir, "new_inference_results.csv")
    if os.path.exists(inference_results_path):
        print(f"Inference results already exist at {inference_results_path}. Skipping inference calculation.")
    else:
        # Calculate inference loss across all data with the final model
        _, all_inference_losses = calculate_inference_loss(model, dataloader, device)
        new_inference_results = [{'window': i + 1, 'inference_loss': loss} for i, loss in enumerate(all_inference_losses)]
        pd.DataFrame(new_inference_results).to_csv(inference_results_path, index=False)
        print(f"New inference results saved to {inference_results_path}")

if __name__ == "__main__":
    main()
