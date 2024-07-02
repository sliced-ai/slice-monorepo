import json
import os
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import datetime
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_NAME = "LAURA"
FOLDER_PATH = '/workspace/slice-monorepo/cl_cr3/aligneddata'
MODEL_NAME = "EleutherAI/pythia-70m"
EPOCHS = 10
BATCH_SIZE = 30
LEARNING_RATE = 1e-4
DATA_SIZE = 15000  # Overall dataset size
RATIO = 0.5  # Ratio for aligneddata and dolly15k
SAVE_FOLDER = "results"

# Function to get next run index
def get_next_run_index(target_name, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    existing_runs = [d for d in os.listdir(save_folder) if d.startswith(target_name)]
    if not existing_runs:
        return 1
    existing_indices = [int(d.split('_')[-1]) for d in existing_runs if d.split('_')[-1].isdigit()]
    return max(existing_indices) + 1 if existing_indices else 1

run_index = get_next_run_index(TARGET_NAME, SAVE_FOLDER)

# Custom Dataset class
class UtteranceDataset(Dataset):
    def __init__(self, utterances, tokenizer, max_length=1000):
        self.utterances = utterances
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        instruction, response = self.utterances[idx]
        text = f"{instruction} {response}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return tokenized

# Function to process folder and extract utterances
def process_folder(folder_path, target_name, num_utterance_pairs):
    utterances = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path) as json_file:
                    data = json.load(json_file)
                    for document in data:
                        turns = document['TURNS']
                        for i in range(1, len(turns)):
                            prev_turn = turns[i-1]
                            curr_turn = turns[i]
                            if target_name in curr_turn['NAMES']:
                                instruction = f"{prev_turn['NAMES'][0]}: " + " ".join(prev_turn['UTTERANCES'])
                                response = f"{curr_turn['NAMES'][0]}: " + " ".join(curr_turn['UTTERANCES'])
                                utterances.append((instruction, response))
                                if len(utterances) >= num_utterance_pairs:
                                    return utterances
    return utterances

# Load and process aligneddata
utterances_aligneddata = process_folder(FOLDER_PATH, TARGET_NAME, DATA_SIZE)
print(f"Total aligneddata pairs loaded: {len(utterances_aligneddata)}")

# Load and process dolly15k data
dataset_dolly15k = load_dataset("databricks/databricks-dolly-15k")['train']
utterances_dolly15k = [(item['instruction'], item['response']) for item in dataset_dolly15k]

print(f"Total dolly15k pairs loaded: {len(utterances_dolly15k)}")

# Function to merge datasets based on a given ratio and overall size
def merge_datasets(aligneddata, dolly15k, ratio, total_size):
    len_aligneddata = int(total_size * ratio)
    len_dolly15k = total_size - len_aligneddata
    merged_data = aligneddata[:len_aligneddata] + dolly15k[:len_dolly15k]
    return merged_data, len_aligneddata, len_dolly15k

# Merging datasets with the specified ratio and overall size
merged_utterances, len_aligneddata, len_dolly15k = merge_datasets(utterances_aligneddata, utterances_dolly15k, RATIO, DATA_SIZE)

print(f"Final merged dataset size: {len(merged_utterances)}")
print(f"Aligneddata size: {len_aligneddata}")
print(f"Dolly15k size: {len_dolly15k}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create dataset
dataset = UtteranceDataset(merged_utterances, tokenizer)

# Model
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# Split dataset
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Collate function for DataLoader
def collate_fn(batch):
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses = []
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    with np.errstate(over='ignore'):
        perplexity = np.exp(avg_train_loss)

    print(f"Ep {epoch+1}/{EPOCHS} | Trn Loss: {avg_train_loss:.4f} | Pplx: {perplexity:.4f}")

end_time = time.time()
total_runtime = end_time - start_time
print(f"Total training time: {total_runtime:.2f} seconds")

# Save model and tokenizer
directory_name = f"{SAVE_FOLDER}/{TARGET_NAME}_{run_index}"
os.makedirs(directory_name, exist_ok=True)
model.save_pretrained(directory_name)
tokenizer.save_pretrained(directory_name)

# Save logs to file
log_file = os.path.join(directory_name, "training_logs.txt")
with open(log_file, "w") as file:
    file.write("Epoch,Train Loss,Perplexity\n")
    for epoch in range(len(train_losses)):
        with np.errstate(over='ignore'):
            perplexity = np.exp(train_losses[epoch])
        file.write(f"{epoch+1},{train_losses[epoch]},{perplexity}\n")

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(directory_name, 'training_plots.png'))
