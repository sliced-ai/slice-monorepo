import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime

# Current date and time for folder naming
now = datetime.datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")
model_name = "gptneo"
directory_name = f"{date_time}-{model_name}"
os.makedirs(directory_name, exist_ok=True)

# Load model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(".")
tokenizer = AutoTokenizer.from_pretrained(".")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("databricks/databricks-dolly-15k")

train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

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

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
model.to('cuda')

def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / torch.numel(correct)
    return accuracy

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(3):
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
        total_train_accuracy += calculate_accuracy(logits, inputs['labels']).item()

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
            total_test_accuracy += calculate_accuracy(logits, inputs['labels']).item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_accuracy)

    print(f"Epoch {epoch + 1}/{3} | "
          f"Trn Loss: {avg_train_loss:.4f} | "
          f"Trn Acc: {avg_train_accuracy:.4f} | "
          f"Tst Loss: {avg_test_loss:.4f} | "
          f"Tst Acc: {avg_test_accuracy:.4f}")

# Save logs to file
log_file = os.path.join(directory_name, "training_logs.txt")
with open(log_file, "w") as file:
    file.write("Epoch,Train Loss,Train Accuracy,Test Loss,Test Accuracy\n")
    for epoch in range(3):
        file.write(f"{epoch+1},{train_losses[epoch]},{train_accuracies[epoch]},{test_losses[epoch]},{test_accuracies[epoch]}\n")

# Save visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 4), train_losses, label='Train Loss')
plt.plot(range(1, 4), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 4), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 4), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.savefig(os.path.join(directory_name, 'training_plots.png'))
