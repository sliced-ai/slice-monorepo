import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MODEL_NAME = "EleutherAI/pythia-70m"

# Set up directory for saving outputs
now = datetime.datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")
directory_name = f"{date_time}-{MODEL_NAME.replace('/', '-')}"
os.makedirs(directory_name, exist_ok=True)

# Load model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load and split dataset
dataset = load_dataset("databricks/databricks-dolly-15k")
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Preprocessing function
def preprocess_function(examples):
    instruction = examples['instruction']
    context = examples['context']
    if isinstance(instruction, list):
        instruction = [" ".join(ins) if isinstance(ins, list) else ins for ins in instruction]
    if isinstance(context, list):
        context = [" ".join(con) if isinstance(con, list) else con for con in context]
    
    text = [f"{ins} {con}" for ins, con in zip(instruction, context)]
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=1000)
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Collate function
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Model to GPU
model.to('cuda')

# Calculate accuracy function
def calculate_accuracy(preds, labels, attention_mask=None):
    preds = torch.argmax(preds, dim=-1)
    if attention_mask is not None:
        mask = attention_mask.bool()
        correct = ((preds == labels) & mask).float()
        accuracy = correct.sum() / mask.sum()
    else:
        correct = (preds == labels).float()
        accuracy = correct.sum() / torch.numel(correct)
    return accuracy

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {key: val.to('cuda') for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        logits = outputs.logits
        total_train_accuracy += calculate_accuracy(logits, inputs['labels'], inputs['attention_mask']).item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    total_test_loss = 0
    total_test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {key: val.to('cuda') for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_test_loss += loss.item()
            
            logits = outputs.logits
            total_test_accuracy += calculate_accuracy(logits, inputs['labels'], inputs['attention_mask']).item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_accuracy)

    perplexity = np.exp(avg_train_loss)
    
    print(f"Ep {epoch+1}/{EPOCHS} | Trn Loss: {avg_train_loss:.4f} | Trn Acc: {avg_train_accuracy:.4f} | Tst Loss: {avg_test_loss:.4f} | Tst Acc: {avg_test_accuracy:.4f} | Pplx: {perplexity:.4f}")

# Save logs to file
log_file = os.path.join(directory_name, "training_logs.txt")
with open(log_file, "w") as file:
    file.write("Epoch,Train Loss,Train Accuracy,Test Loss,Test Accuracy,Perplexity\n")
    for epoch in range(EPOCHS):
        file.write(f"{epoch+1},{train_losses[epoch]},{train_accuracies[epoch]},{test_losses[epoch]},{test_accuracies[epoch]},{np.exp(train_losses[epoch])}\n")

# Save visualizations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, EPOCHS+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.savefig(os.path.join(directory_name, 'training_plots.png'))
