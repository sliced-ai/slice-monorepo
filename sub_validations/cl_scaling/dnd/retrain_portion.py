import os
import json
import torch
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gc
from glob import glob
from itertools import islice

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
            'attention_mask': tokens['attention_mask'].squeeze()
        }

def clean_mem(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

def retrain_model(cfg, data, start_batch, end_batch):
    device = torch.device("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader for the entire dataset
    full_dataset = QADataset(data, tokenizer, max_len=cfg["max_length"])
    full_dataloader = DataLoader(full_dataset, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    # Select the specified batch range from the dataloader
    selected_batches = list(islice(full_dataloader, start_batch, end_batch))
    
    # Create a DataLoader for the selected batches
    dataloader = DataLoader(selected_batches, batch_size=1, shuffle=False, pin_memory=True)

    # Create a separate directory for the retraining results
    experiment_dir = os.path.join('experiments', cfg['experiment_name'])
    retrain_dir = os.path.join(experiment_dir, f"retrain_{start_batch}_{end_batch}")
    os.makedirs(retrain_dir, exist_ok=True)

    # Check if the retrained model already exists
    retrained_model_path = os.path.join(retrain_dir, "retrained_model.pt")
    if os.path.exists(retrained_model_path):
        print(f"Retrained model already exists at {retrained_model_path}. Skipping retraining.")
        return retrained_model_path

    # Load the last saved model
    final_model_path = os.path.join(experiment_dir, cfg["main_model"]["save_dir"], "final_model.pt")
    model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
    model.load_state_dict(torch.load(final_model_path))
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["fixed_learning_rate"])

    # Retrain on the specified batches for a single epoch
    model.train()
    for batch in dataloader:
        batch_inputs = {k: v[0].to(device, non_blocking=True) for k, v in batch.items()}
        opt.zero_grad()
        outputs = model(**batch_inputs, labels=batch_inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        opt.step()

    # Save the retrained model
    torch.save(model.state_dict(), retrained_model_path)

    clean_mem(model)
    return retrained_model_path

def calculate_inference_loss(model, dataloader):
    device = torch.device("cuda:1")
    model.to(device)
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
    config_path = 'config.json'
    cfg = load_config(config_path)
    data_dir = cfg['data_dir']
    experiment_name = cfg['experiment_name']
    
    data = load_json_files(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = QADataset(data, tokenizer, max_len=cfg["max_length"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    # Retrain the model on batches 3000-4000
    retrained_model_path = retrain_model(cfg, data, 3000, 4000)

    # Load the retrained model
    model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])
    model.load_state_dict(torch.load(retrained_model_path))
    model.to(torch.device("cuda:0"))

    # Perform inference on all batches again
    retrain_dir = os.path.join('experiments', experiment_name, f"retrain_3000_4000")
    inference_results_path = os.path.join(retrain_dir, "retrained_inference_results.csv")
    if os.path.exists(inference_results_path):
        print(f"Inference results already exist at {inference_results_path}. Skipping inference calculation.")
    else:
        _, all_inference_losses = calculate_inference_loss(model, dataloader)
        new_inference_results = [{'batch': i + 1, 'inference_loss': loss} for i, loss in enumerate(all_inference_losses)]
        pd.DataFrame(new_inference_results).to_csv(inference_results_path, index=False)
        print(f"New inference results saved to {inference_results_path}")

if __name__ == "__main__":
    main()
