import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import json
from torch.cuda.amp import autocast
import bitsandbytes as bnb
import re

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")

# Check if tokenizer has a padding token, if not, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id,token="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")

def extract_actual_number_from_labels(labels):
    """Assumes the actual number is the last token in the labels."""
    actual_numbers = labels[:, -1]  # Extract the last token
    return actual_numbers


def extract_predicted_number(output):
    # Convert logits to probabilities if necessary
    if isinstance(output, torch.Tensor):  
        probabilities = torch.softmax(output, dim=0)  
    else: 
        probabilities = output

    # Most likely token
    predicted_token_id = probabilities.argmax().item() 
    predicted_token = tokenizer.decode(predicted_token_id)
    #print(predicted_token)
    # Try to extract a number from the predicted token 
    match = re.search(r"\d+", predicted_token)
    if match:
        return int(match.group())
    else:
        return None  

def calculate_average_with_extraction(outputs):
    predicted_numbers = [extract_predicted_number(output)
                         for output in outputs.logits[:, -1, :]]

    # Filter out None values (where no digits were extracted)
    predicted_numbers = [num for num in predicted_numbers if num is not None]

    if predicted_numbers:
        return torch.tensor(predicted_numbers).float().mean().item()
    else:
        return None  # Return None if no valid predictions in the batch


def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        for _ in range(batch_size):  # Repeat for the batch size
            prompt = f"{base_prompt}" 
            completion = f"{i}"  # Use i directly for number prediction
            dataset.append({"prompt": prompt, "completion": completion}) 
    return dataset

# User input
max_number = 10000
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

for epoch in range(num_epochs):
    model.train()
    current_expected_value = 0
    for batch in train_dataloader:
        input_ids = torch.tensor([item['input_ids'] for item in batch]).to(device)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch]).to(device)
        labels = torch.tensor([item['labels'] for item in batch]).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        print(f"Loss: {loss.item()}")
        average_predicted = calculate_average_with_extraction(outputs)
        #if average_predicted is not None:
            #print(f"Average Predicted: {average_predicted}, Expected Value: {current_expected_value}")
        #print(f"Average Predicted: {average_predicted}, Average Actual: {current_expected_value}")
        current_expected_value+=1
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
model.save_pretrained("./model_output")