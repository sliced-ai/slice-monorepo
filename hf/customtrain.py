import json
import re
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb

# Constants
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda")

def load_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, token="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")
    return tokenizer, model

def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        for _ in range(batch_size):
            prompt = f"{base_prompt}"
            completion = f"{i}"
            dataset.append({"prompt": prompt, "completion": completion})
    return dataset

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    # Using the tokenizer as target tokenizer to process completions
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)['input_ids']

    labels_padded = []
    for label, input_id in zip(labels, model_inputs['input_ids']):
        # Calculate padding length and create a padded label list
        padding_length = len(input_id) - len(label)
        label_padded = label + [-100] * padding_length
        labels_padded.append(label_padded)

    model_inputs['labels'] = labels_padded

    return model_inputs

def mse_loss_function(predicted_numbers, actual_number, device):
    if predicted_numbers:
        numbers_tensor = torch.tensor(predicted_numbers, dtype=torch.float, device=device, requires_grad=True)
        actual_numbers_tensor = torch.full_like(numbers_tensor, actual_number, dtype=torch.float, requires_grad=True)
        mse_loss = torch.mean((numbers_tensor - actual_numbers_tensor) ** 2)
        return mse_loss
    else:
        return torch.tensor(0.1, dtype=torch.float, device=device, requires_grad=True)

def penalty_loss_function(outputs, expected_value, device):
    dummy_prediction = torch.tensor([expected_value], dtype=torch.float, device=device, requires_grad=True)
    outputs_mean = outputs.logits.mean(dim=2)
    outputs_mean = outputs_mean.mean(dim=1, keepdim=True)
    penalty_loss = torch.mean((outputs_mean - dummy_prediction) ** 2)
    return penalty_loss

def extract_predicted_number(logits, tokenizer):
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_ids = probabilities.argmax(dim=-1)
    predicted_tokens = [tokenizer.decode([token_id]) for token_id in predicted_token_ids]
    print(predicted_tokens[:-1])
    extracted_numbers = []
    for predicted_token in predicted_tokens:
        match = re.search(r"\d+", predicted_token)
        if match:
            extracted_numbers.append(int(match.group()))
        else:
            extracted_numbers.append(None)
    return extracted_numbers

def calculate_average_with_extraction(outputs, actual_number, tokenizer):
    predicted_numbers = [extract_predicted_number(single_logits, tokenizer) for single_logits in outputs.logits]
    flat_list = [num for sublist in predicted_numbers for num in sublist if num is not None]
    print(f"Flattened and filtered numbers: {flat_list}")
    if flat_list:
        average = sum(flat_list) / len(flat_list)
        return [average], actual_number
    else:
        return None, actual_number

def train_model(model, train_dataloader, optimizer, tokenizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        current_expected_value = 0
        for batch in train_dataloader:
            
            input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch]).to(DEVICE)
            attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch]).to(DEVICE)
            labels = torch.stack([torch.tensor(item['labels']) for item in batch]).to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            print("TRAIN: ")
            logits = outputs.logits
            predicted_ids = logits.argmax(dim=-1)
            
            # Print the input prompt
            input_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print("Input Prompt:", input_prompt)
            
            # Print the predicted tokens
            predicted_tokens = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print("Predicted Tokens:", predicted_tokens)
            stop

            
            predicted_numbers, actual_number = calculate_average_with_extraction(outputs, current_expected_value, tokenizer)

            if predicted_numbers and current_expected_value > 200:
                loss = mse_loss_function(predicted_numbers, current_expected_value, DEVICE)
            else:
                loss = penalty_loss_function(outputs, current_expected_value, DEVICE)

            optimizer.zero_grad()
            if loss is not None:
                loss.backward()
                optimizer.step()

            if predicted_numbers:
                average_predicted = torch.tensor([num for num in predicted_numbers if num is not None], dtype=torch.float).mean().item()
                print(f"Average Predicted: {average_predicted}, Actual Value: {actual_number}, Loss: {loss.item()}")
            else:
                print(f"No numbers predicted, Loss: {loss.item()}")

            current_expected_value += 1

    return model

def main():
    print("STARTING CL COUNTING TESTS")
    max_number = 1000
    batch_size = 1
    base_prompt = f"You are learning to count. Please remember your previous number and respond with an integrer of what is the next number.Only respond with a single number.Your next number prediction is:"

    num_epochs = 1

    tokenizer, model = load_tokenizer_and_model(MODEL_ID)
    train_data = generate_dataset(base_prompt, max_number, batch_size)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)                 
    
    # Tokenize the updated prompt
    inputs = tokenizer(base_prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    model.to(DEVICE)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("INFERENCE: ")
    print("Generated text:", generated_text)

    
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.0001)
    model.to(DEVICE)

    trained_model = train_model(model, train_dataloader, optimizer, tokenizer, num_epochs)
    trained_model.save_pretrained("./model_output")

if __name__ == "__main__":
    main()