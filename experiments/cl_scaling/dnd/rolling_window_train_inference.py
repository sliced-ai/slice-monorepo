import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Define the configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-70m"
    },
    "data_dir": "/workspace/slice-monorepo/cl_cr3/aligneddata_final_cleaned",
    "experiment_name": "rw_7",
    "fixed_learning_rate": 5e-07,
    "window_size": 2049,
    "step_size": 500,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "experiments_dir": "/workspace/slice-monorepo/sub_validations/cl_scaling/dnd/experiments",
    "num_epochs": 10
}

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

def save_config(cfg, experiment_dir):
    config_path = os.path.join(experiment_dir, 'rolling_window_config.json')
    with open(config_path, 'w') as cfg_file, open('rolling_window_config.json', 'w') as local_cfg_file:
        json.dump(cfg, cfg_file, indent=4)
        json.dump(cfg, local_cfg_file, indent=4)
    return config_path

def ensure_dirs(cfg):
    exp_dir = os.path.join(cfg["experiments_dir"], cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

class Trainer:
    def __init__(self, cfg, data_loader):
        self.cfg = cfg
        self.dl = data_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["fixed_learning_rate"])
        self.num_epochs = cfg["num_epochs"]
        self.results = []

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for i, batch in enumerate(self.dl):
                batch_inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Step {i + 1}/{len(self.dl)}: Train Loss = {loss.item()}")
                self.results.append({'epoch': epoch + 1, 'step': i + 1, 'loss': loss.item()})
            print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(self.dl)}")

            # Save the final model at the end of training
            final_model_path = os.path.join(self.cfg["experiment_dir"], f"{self.cfg['experiment_name']}_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved at: {final_model_path}")

        # Save the training loss results to a CSV file
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(self.cfg["experiment_dir"], "training_loss.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Training loss saved to: {results_csv_path}")

def calculate_inference_loss(model, dataloader, device):
    model.eval()
    all_inference_losses = []

    with torch.no_grad():
        for batch in dataloader:
            batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
            loss = outputs.loss
            all_inference_losses.append(loss.item())

    return all_inference_losses

def main():
    experiment_dir = ensure_dirs(cfg)
    cfg["experiment_dir"] = experiment_dir  # Add experiment_dir to cfg for convenience
    save_config(cfg, experiment_dir)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"])
    
    dataset = RollingWindowDataset('tokenized_utterances.pt', window_size=cfg["window_size"], step_size=cfg["step_size"])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    num_steps = len(dataset)
    print(f"Total training steps: {num_steps}")

    final_model_path = os.path.join(experiment_dir, f"{cfg['experiment_name']}.pt")
    if os.path.exists(final_model_path):
        print(f"Model already exists at {final_model_path}. Skipping training and proceeding to inference.")
        model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
        model.load_state_dict(torch.load(final_model_path, map_location=cfg["gpu_device"]))
        model.to(cfg["gpu_device"])
    else:
        trainer = Trainer(cfg, data_loader)
        trainer.train()
        model = trainer.model  # Use the trained model for inference

    # Inference
    inference_results_path = os.path.join(experiment_dir, "new_inference_results.csv")
    if os.path.exists(inference_results_path):
        print(f"Inference results already exist at {inference_results_path}. Skipping inference calculation.")
    else:
        # Calculate inference loss across all data with the final model
        all_inference_losses = calculate_inference_loss(model, data_loader, cfg["gpu_device"])
        new_inference_results = [{'window': i + 1, 'inference_loss': loss} for i, loss in enumerate(all_inference_losses)]
        pd.DataFrame(new_inference_results).to_csv(inference_results_path, index=False)
        print(f"New inference results saved to {inference_results_path}")

if __name__ == "__main__":
    main()
