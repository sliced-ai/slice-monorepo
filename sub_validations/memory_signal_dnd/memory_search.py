import os
import json
import torch
import pandas as pd
import gc
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Main Configuration embedded in the script
main_config = {
    "experiment_base_dir": "experiments",
    "experiment_name": "memory_rw_3",
    "gpu_device": "cuda:2",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "tokenized_utterances_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/dnd/tokenized_utterances.pt",
    "data_dirs": ["rw_3"],
    "retrain_percentages": [1,5, 10],  # List of retraining percentages
    "number_of_retrain": 4,  # Number of retraining points
    "inference_extension_percentage": 5,  # Percentage of total size to extend inference range
    "learning_rates": [1e-4, 1e-5, 1e-6]  # List of learning rates to experiment with
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
        if end_idx > len(self.tokens):
            raise IndexError
        token_sequence = self.tokens[start_idx:end_idx]
        return {
            'input_ids': torch.tensor(token_sequence, dtype=torch.long)
        }

def save_config(cfg, experiment_dir):
    config_path = os.path.join(experiment_dir, 'memory_search.json')
    os.makedirs(experiment_dir, exist_ok=True)
    with open(config_path, 'w') as cfg_file:
        json.dump(cfg, cfg_file, indent=4)
    print(f"Config file saved to {config_path}")

def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()

def calculate_loss(model, dataloader, device, start_window):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_losses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            all_losses.append((start_window + i, loss.item()))

    return total_loss / num_batches, all_losses

def load_folder_config(data_dir):
    config_path = os.path.join(data_dir, 'rolling_window_config.json')
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def calculate_ranges(total_samples, retrain_percentage, number_of_retrain, inference_extension_percentage):
    num_samples = int(total_samples * (retrain_percentage / 100.0))
    retrain_ranges = []
    inference_ranges = []

    step = total_samples // number_of_retrain
    inference_extension = int(total_samples * (inference_extension_percentage / 100.0))

    for i in range(number_of_retrain):
        retrain_center = step * (i + 0.5)
        start_window = max(0, int(retrain_center - num_samples // 2))
        end_window = min(total_samples, int(retrain_center + num_samples // 2))

        inference_start_window = max(0, start_window - inference_extension)
        inference_end_window = min(total_samples, end_window + inference_extension)

        retrain_ranges.append((start_window, end_window))
        inference_ranges.append((inference_start_window, inference_end_window))

    return retrain_ranges, inference_ranges

def print_experiment_details(data_dir, model_name, model_path, total_samples, retrain_ranges, inference_ranges, learning_rate, retrain_percentage):
    for i in range(len(retrain_ranges)):
        print(f"\nExperiment {i+1} for Data Directory: {data_dir}")
        print(f"Model Path: {model_path}")
        print(f"Base Model: {model_name}")
        print(f"Total Samples: {total_samples}")
        print(f"Retrain Range: {retrain_ranges[i][0]} - {retrain_ranges[i][1]}")
        print(f"Inference Range: {inference_ranges[i][0]} - {inference_ranges[i][1]}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Retrain Percentage: {retrain_percentage}\n")

def check_data_lengths(data_dir, total_samples):
    training_loss_path = os.path.join(data_dir, 'training_loss.csv')
    inference_results_path = os.path.join(data_dir, 'new_inference_results.csv')

    training_loss_df = pd.read_csv(training_loss_path)
    inference_results_df = pd.read_csv(inference_results_path)

    training_length = len(training_loss_df)
    inference_length = len(inference_results_df)

    print(f"Total Samples: {total_samples}")
    print(f"Training CSV Length: {training_length}")
    print(f"Inference CSV Length: {inference_length}")

    assert total_samples == training_length == inference_length, "Mismatch in data lengths!"

def run_experiment(base_model_name, model_weights_path, tokenizer, main_cfg, data_dir, retrain_range, inference_range, results, training_results, learning_rate, retrain_percentage):
    folder_cfg = load_folder_config(data_dir)
    window_size = folder_cfg["window_size"]
    step_size = folder_cfg["step_size"]

    device = torch.device(main_cfg["gpu_device"])

    token_file = main_cfg['tokenized_utterances_path']
    dataset = RollingWindowDataset(token_file, window_size=window_size, step_size=step_size)

    start_window, end_window = retrain_range
    inference_start_window, inference_end_window = inference_range

    # Training phase
    selected_indices = list(range(start_window, end_window))
    selected_dataset = torch.utils.data.Subset(dataset, selected_indices)
    train_dataloader = DataLoader(selected_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Load the base model and apply the local weights
    model = GPTNeoXForCausalLM.from_pretrained(base_model_name).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for i, batch in enumerate(train_dataloader):
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        training_results.append({
            'folder': data_dir,
            'retrain_window': f"{start_window}-{end_window}",
            'window': start_window + i,
            'training_loss': loss.item(),
            'learning_rate': learning_rate,
            'retrain_percentage': retrain_percentage
        })

    # Inference phase
    inference_indices = list(range(inference_start_window, inference_end_window))
    inference_dataset = torch.utils.data.Subset(dataset, inference_indices)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, pin_memory=True)

    avg_loss, all_inference_losses = calculate_loss(model, inference_dataloader, device, inference_start_window)
    for window, loss in all_inference_losses:
        results.append({
            'folder': data_dir,
            'retrain_window': f"{start_window}-{end_window}",
            'inference_window': f"{inference_start_window}-{inference_end_window}",
            'window': window,
            'inference_loss': loss,
            'learning_rate': learning_rate,
            'retrain_percentage': retrain_percentage
        })

    # Clear memory
    del model
    clean_mem()

def main():
    cfg = main_config  # Use the embedded config
    experiment_name = cfg['experiment_name']
    experiment_dir = os.path.join(cfg["experiment_base_dir"], experiment_name)
    
    # Save the main config file at the start of the experiment
    save_config(cfg, experiment_dir)

    results = []
    training_results = []

    device = torch.device(cfg["gpu_device"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg['tokenizer_path'], clean_up_tokenization_spaces=True)

    for data_dir in cfg["data_dirs"]:
        folder_cfg = load_folder_config(data_dir)
        model_name = folder_cfg["main_model"]["name"]
        model_path = os.path.join(data_dir, f'{data_dir}_model.pt')
        
        token_file = cfg['tokenized_utterances_path']
        dataset = RollingWindowDataset(token_file, window_size=folder_cfg["window_size"], step_size=folder_cfg["step_size"])
        total_samples = len(dataset)
        
        check_data_lengths(data_dir, total_samples)
        
        for retrain_percentage in cfg["retrain_percentages"]:
            retrain_ranges, inference_ranges = calculate_ranges(total_samples, retrain_percentage, cfg['number_of_retrain'], cfg['inference_extension_percentage'])
            
            for learning_rate in cfg["learning_rates"]:
                print_experiment_details(data_dir, model_name, model_path, total_samples, retrain_ranges, inference_ranges, learning_rate, retrain_percentage)
                
                for retrain_range, inference_range in zip(retrain_ranges, inference_ranges):
                    run_experiment(model_name, model_path, tokenizer, cfg, data_dir, retrain_range, inference_range, results, training_results, learning_rate, retrain_percentage)

                # Save the results to a CSV file for inference after each learning rate change
                results_df = pd.DataFrame(results)
                results_csv_path = os.path.join(experiment_dir, 'retrained_inference_results.csv')
                if os.path.exists(results_csv_path):
                    results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
                else:
                    results_df.to_csv(results_csv_path, index=False)
                print(f"Inference results saved to {results_csv_path}")

                # Save the results to a CSV file for training after each learning rate change
                training_results_df = pd.DataFrame(training_results)
                training_results_csv_path = os.path.join(experiment_dir, 'retrained_training_results.csv')
                if os.path.exists(training_results_csv_path):
                    training_results_df.to_csv(training_results_csv_path, mode='a', header=False, index=False)
                else:
                    training_results_df.to_csv(training_results_csv_path, index=False)
                print(f"Training results saved to {training_results_csv_path}")

if __name__ == "__main__":
    main()
