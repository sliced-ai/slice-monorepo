import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

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

def save_config(cfg, experiment_dir):
    config_path = os.path.join(experiment_dir, 'rolling_window_config.json')
    with open(config_path, 'w') as cfg_file:
        json.dump(cfg, cfg_file, indent=4)
    return config_path

def ensure_dirs(cfg):
    exp_dir = os.path.join('experiments', cfg["experiment_name"])
    cfg["main_model"]["save_dir"] = os.path.join(exp_dir, cfg["main_model"]["save_dir"])
    os.makedirs(cfg["main_model"]["save_dir"], exist_ok=True)
    return exp_dir

class Trainer:
    def __init__(self, cfg, data_loader, num_steps):
        self.cfg = cfg
        self.dl = data_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["fixed_learning_rate"])
        self.num_steps = num_steps
        self.results = []

    def train(self):
        self.model.train()
        for i, batch in enumerate(self.dl):
            batch_inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f"Processed Window {i + 1}/{self.num_steps}: Train Loss = {loss.item()}")
            self.results.append({'window': i + 1, 'loss': loss.item()})

        # Save the final model at the end of training
        final_model_path = os.path.join(self.cfg["main_model"]["save_dir"], "final_model.pt")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved at: {final_model_path}")

        # Save the training loss results to a CSV file
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(self.cfg["main_model"]["save_dir"], "training_loss.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Training loss saved to: {results_csv_path}")

def main():
    cfg = load_config('rolling_window_config.json')
    experiment_dir = ensure_dirs(cfg)
    save_config(cfg, experiment_dir)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"])
    
    dataset = RollingWindowDataset('tokenized_utterances.pt', window_size=cfg["window_size"], step_size=cfg["step_size"])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    # Example data printout for verification
    example_window = dataset[0]
    example_tokens = example_window['input_ids'].tolist()
    example_text = tokenizer.decode(example_tokens)
    print(f"Example Tokens: {example_tokens[:50]}")  # Print first 50 tokens for brevity
    print(f"Decoded Text: {example_text[:500]}")  # Print first 500 characters for brevity
    
    num_steps = len(dataset)
    print(f"Total training steps: {num_steps}")

    trainer = Trainer(cfg, data_loader, num_steps)
    trainer.train()

if __name__ == "__main__":
    main()
