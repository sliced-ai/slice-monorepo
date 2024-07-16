import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np

# Constants
EPOCHS = 2
BATCH_SIZE = 30
LEARNING_RATE = 1e-5  # Reduced learning rate
DEVICE = 'cuda:0'
MODEL_NAME = "EleutherAI/pythia-70m"
MAX_INPUT_LENGTH = 64  # Maximum length for input prompts

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
    tokenized_inputs = tokenizer(inputs, truncation=True, padding=False)
    
    # Filter out examples with input length exceeding the maximum allowed length
    filtered_indices = [i for i, ids in enumerate(tokenized_inputs['input_ids']) if len(ids) <= MAX_INPUT_LENGTH]
    filtered_inputs = [inputs[i] for i in filtered_indices]
    filtered_responses = [response[i] for i in filtered_indices]

    tokenized_inputs = tokenizer(filtered_inputs, truncation=True, padding='max_length', max_length=1000)
    labels = tokenizer(filtered_responses, truncation=True, padding='max_length', max_length=1000).input_ids

    labels = [label + [tokenizer.pad_token_id] * (1000 - len(label)) for label in labels]
    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    tokenized_inputs['labels'] = labels.tolist()
    return tokenized_inputs

# Preprocess the training dataset
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

# Print the length of the new filtered dataset
print(f"Length of the new filtered dataset: {len(train_dataset)}")

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Create DataLoader for training
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initial inference on 2 examples
print("\nInitial Inference on 2 examples from the training set:")
model.to(DEVICE)
model.eval()

for i in range(2):
    example = train_dataset[i]
    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(DEVICE)
    attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(DEVICE)
    labels = torch.tensor(example['labels']).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100)

    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, "").strip()
    expected_output_text = tokenizer.decode([id for id in labels[0].cpu().numpy() if id != -100], skip_special_tokens=True)

    print(f"Input: {input_text}")
    print(f"Expected Output: {expected_output_text}")
    print(f"Model Output: {output_text}\n")

# Clear model and data from GPU
model.cpu()
torch.cuda.empty_cache()

# Re-load model to GPU for training
model.to(DEVICE)
model.train()

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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
    perplexity = np.exp(avg_train_loss)

    print(f"Ep {epoch + 1}/{EPOCHS} | Trn Loss: {avg_train_loss:.4f} | Pplx: {perplexity:.4f}")

# Final inference on 2 examples
print("\nFinal Inference on 2 examples from the training set:")
model.eval()

for i in range(2):
    example = train_dataset[i]
    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(DEVICE)
    attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(DEVICE)
    labels = torch.tensor(example['labels']).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100)

    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, "").strip()
    expected_output_text = tokenizer.decode([id for id in labels[0].cpu().numpy() if id != -100], skip_special_tokens=True)

    print(f"Input: {input_text}")
    print(f"Expected Output: {expected_output_text}")
    print(f"Model Output: {output_text}\n")
