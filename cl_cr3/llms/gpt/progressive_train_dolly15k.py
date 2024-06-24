import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

# Constants
EPOCHS = 5
BATCH_SIZE = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda:0'
MODEL_NAME = "EleutherAI/pythia-70m"

def get_run_number(target_name):
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        return 1
    existing_runs = [d for d in os.listdir(result_dir) if d.startswith(target_name)]
    numeric_parts = [int(d.split('_')[-1]) for d in existing_runs if d.split('_')[-1].isdigit()]
    if not numeric_parts:
        return 1
    latest_run = max(numeric_parts)
    return latest_run + 1

# Get run number
run_number = get_run_number(MODEL_NAME.replace('/', '-'))
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
directory_name = f"results/{MODEL_NAME.replace('/', '-')}_{run_number}"
os.makedirs(directory_name, exist_ok=True)

# Load model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Split dataset into train and test
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']

def preprocess_function(examples):
    instruction = examples['instruction']
    context = examples.get('context', "")
    response = examples['response']

    inputs = [f"{ins} {con}".strip() for ins, con in zip(instruction, context)]
    inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=1000)
    
    labels = tokenizer(response, truncation=True, padding='max_length', max_length=1000).input_ids
    labels = [label + [tokenizer.pad_token_id] * (1000 - len(label)) for label in labels]

    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    inputs['labels'] = labels
    return inputs

# Preprocess the training dataset
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Create DataLoader for training
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.train()
model.to(DEVICE)

train_losses = []

# Track start time
start_time = datetime.datetime.now()

# Training loop
for epoch in range(EPOCHS):
    total_train_loss = 0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {key: val.to(DEVICE) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    perplexity = np.exp(avg_train_loss)

    print(f"Ep {epoch + 1}/{EPOCHS} | Trn Loss: {avg_train_loss:.4f} | Pplx: {perplexity:.4f}")

# Calculate total training time
total_time = (datetime.datetime.now() - start_time).total_seconds()
print(f"Total training time: {total_time:.2f} seconds")

# Save logs to file
log_file = os.path.join(directory_name, "training_logs.txt")
with open(log_file, "w") as file:
    file.write("Epoch,Train Loss,Perplexity\n")
    for epoch in range(EPOCHS):
        file.write(f"{epoch+1},{train_losses[epoch]},{np.exp(train_losses[epoch])}\n")

# Save model and tokenizer
model.save_pretrained(directory_name)
tokenizer.save_pretrained(directory_name)

# Save training loss plot
plt.figure(figsize=(12, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(directory_name, 'training_loss.png'))
