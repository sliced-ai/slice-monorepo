import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import json
from torch.cuda.amp import autocast
import bitsandbytes as bnb
import re
import torch.nn as nn

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check if tokenizer has a padding token, if not, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id,token="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")

def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        for _ in range(batch_size):  # Repeat for the batch size
            prompt = f"{base_prompt}" 
            completion = f"{i}"  # Use i directly for number prediction
            dataset.append({"prompt": prompt, "completion": completion}) 
    return dataset

# User input
max_number = 100
batch_size = 8
base_prompt = f"You are training in order to guess the next number as we count. I have been training you on an increasing number count from 0 to {max_number} in integer steps. Please remember previous inputs to training and guess the next number in the sequence. Assume you have started at zero. You will only respond with your predicted next integer. What is your next number? "

# Generate the dataset
train_data = generate_dataset(base_prompt, max_number, batch_size)

def tokenize_function(examples):
    # Tokenize both prompts and completions
    model_inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    # Prepare labels which are aligned with the prompts: labels should be the continuation of the prompts
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)['input_ids']
    # Pad labels to match the length of model inputs
    labels_padded = []
    for label, input_id in zip(labels, model_inputs['input_ids']):
        label_padded = label + [-100] * (len(input_id) - len(label))
        labels_padded.append(label_padded)
    model_inputs['labels'] = labels_padded
    return model_inputs

train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Create a DataLoader for training
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

# Set up the optimizer and learning rate
#optimizer = torch.optim.AdamW_bnb_8bit(model.parameters(), lr=2e-5)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001) # instead of torch.optim.Adam

# Custom training loop
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def mse_loss_function(predicted_numbers, actual_number, device):
    # Convert numbers to a tensor with requires_grad
    if predicted_numbers:
        numbers_tensor = torch.tensor(predicted_numbers, dtype=torch.float, device=device, requires_grad=True)
        actual_numbers_tensor = torch.full_like(numbers_tensor, actual_number, dtype=torch.float, requires_grad=True)
        mse_loss = torch.mean((numbers_tensor - actual_numbers_tensor) ** 2)
        return mse_loss
    else:
        # If no numbers to calculate loss, return a small constant loss to keep the graph
        return torch.tensor(0.1, dtype=torch.float, device=device, requires_grad=True)

def penalty_loss_function(outputs, expected_value, device):
    # Simplified penalty loss when no number is predicted
    dummy_prediction = torch.tensor([expected_value], dtype=torch.float, device=device, requires_grad=True)
    outputs_mean = outputs.logits.mean(dim=2)  # Assuming logits are [batch_size, sequence_length, vocab_size]
    outputs_mean = outputs_mean.mean(dim=1, keepdim=True)  # Reduce to mean per batch
    penalty_loss = torch.mean((outputs_mean - dummy_prediction) ** 2)  # Simplified penalty loss
    return penalty_loss

def calculate_average_with_extraction(outputs, actual_number):
    # Assuming 'outputs' has an attribute 'logits'
    # Extract numbers from each batch of logits
    predicted_numbers = [extract_predicted_number(single_logits) for single_logits in outputs.logits]

    # Flatten the list of lists and filter out None values
    flat_list = [num for sublist in predicted_numbers for num in sublist if num is not None]

    print(f"Flattened and filtered numbers: {flat_list}")

    # Calculate average if there are any numbers, otherwise return None
    if flat_list:
        average = sum(flat_list) / len(flat_list)
        return [average], actual_number  # Return as a list for consistency
    else:
        return None, actual_number


def extract_predicted_number(logits):
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_ids = probabilities.argmax(dim=-1)
    predicted_tokens = [tokenizer.decode([token_id]) for token_id in predicted_token_ids]
    extracted_numbers = []
    for predicted_token in predicted_tokens:
        match = re.search(r"\d+", predicted_token)
        if match:
            extracted_numbers.append(int(match.group()))
        else:
            extracted_numbers.append(None)
    return extracted_numbers

for epoch in range(num_epochs):
    model.train()
    current_expected_value = 0
    for batch in train_dataloader:
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch]).to(device)
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch]).to(device)
        labels = torch.stack([torch.tensor(item['labels']) for item in batch]).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        predicted_numbers, actual_number = calculate_average_with_extraction(outputs, current_expected_value)
        
        if predicted_numbers:
            loss = mse_loss_function(predicted_numbers, current_expected_value, device)
        else:
            loss = penalty_loss_function(outputs, current_expected_value, device)
        
        optimizer.zero_grad()
        if loss is not None:
            loss.backward()
            optimizer.step()

        # Output for monitoring
        print(f"Epoch {epoch+1}, Batch {current_expected_value+1}:")
        print(f"input_ids shape {input_ids.shape}")
        print(f"attention_mask shape {attention_mask.shape}")
        print(f"labels shape {labels.shape}")
        print(f"output logits shape {outputs.logits.shape}")
        if predicted_numbers:
            average_predicted = torch.tensor([num for num in predicted_numbers if num is not None], dtype=torch.float).mean().item()
            print(f"Average Predicted: {average_predicted}, Actual Value: {actual_number}, Loss: {loss.item()}")
        else:
            print(f"No numbers predicted, Loss: {loss.item()}")

        current_expected_value += 1


# Save the trained model
model.save_pretrained("./model_output")