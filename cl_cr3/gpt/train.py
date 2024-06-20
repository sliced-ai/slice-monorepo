import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(".")
tokenizer = AutoTokenizer.from_pretrained(".")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Split dataset into training and test sets
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Preprocess dataset
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

# Apply the preprocessing function to the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Define a collate function to convert lists to tensors
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
model.to('cuda')

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / torch.numel(correct)
    return accuracy

# Lists to store metrics
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(3):  # Number of epochs
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

# Plotting the training metrics
epochs = range(1, 4)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
