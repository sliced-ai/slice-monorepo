import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import random
import time

# Configuration
INITIAL_SIZE = 80  # Initial dataset size
INCREMENT_RATIO = 0.1  # Increment ratio for each step
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
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
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Model to GPU
model.to('cuda')

# Progressive training loop
def progressive_training(model, train_dataset, initial_size, increment_ratio, max_epochs, device):
    total_size = len(train_dataset)
    train_size = int((1 - 0.1) * total_size)
    test_size = total_size - train_size

    train_dataset, _ = random_split(train_dataset, [train_size, test_size])

    current_size = initial_size
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(max_epochs):
        if current_size > train_size:
            break

        current_train_indices = torch.randperm(train_size)[:int(current_size)]
        current_train_dataset = torch.utils.data.Subset(train_dataset, current_train_indices)
        #current_train_dataset = torch.utils.data.Subset(train_dataset, list(range(int(current_size))))
        train_dataloader = DataLoader(current_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

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

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)

        perplexity = np.exp(avg_train_loss)

        print(f"Ep {epoch+1}/{max_epochs} | Data size: {current_size} | Trn Loss: {avg_train_loss:.4f} | Tst Loss: {avg_test_loss:.4f} | Pplx: {perplexity:.4f}")

        current_size += int(current_size * increment_ratio)

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total training time: {total_runtime:.2f} seconds")

    return train_losses, test_losses

# Training with progressive data growth
train_losses, test_losses = progressive_training(
    model, train_dataset, INITIAL_SIZE, INCREMENT_RATIO, EPOCHS, 'cuda'
)

# Save logs to file
log_file = os.path.join(directory_name, "training_logs.txt")
with open(log_file, "w") as file:
    file.write("Epoch,Train Loss,Test Loss,Perplexity\n")
    for epoch in range(len(train_losses)):
        file.write(f"{epoch+1},{train_losses[epoch]},{test_losses[epoch]},{np.exp(train_losses[epoch])}\n")

# Save visualizations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.savefig(os.path.join(directory_name, 'training_plots.png'))

# Final test with 5 random inputs from test set
print("\nFinal Test on 5 random inputs from the test set:")
model.eval()
random_indices = random.sample(range(len(test_dataset)), 5)
test_results = []
for idx in random_indices:
    input_data = test_dataset[idx]
    input_ids = torch.tensor(input_data['input_ids']).unsqueeze(0).to('cuda')
    attention_mask = torch.tensor(input_data['attention_mask']).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1000)
    
    input_text = tokenizer.decode(input_data['input_ids'], skip_special_tokens=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    test_results.append({"input": input_text, "output": output_text})
    print(f"Input: {input_text}")
    print(f"Output: {output_text}\n")

# Save test results to file
test_results_file = os.path.join(directory_name, "test_results.txt")
with open(test_results_file, "w") as file:
    for result in test_results:
        file.write(f"Input: {result['input']}\n")
        file.write(f"Output: {result['output']}\n\n")
