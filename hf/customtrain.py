import json
import re
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
from num2words import num2words
from word2number import w2n
import spacy
import re
from word2number import w2n
import random
nlp = spacy.load("en_core_web_sm")

# Constants
#MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda")

def log_message(batch_idx, loss, unique_numbers, idx, input_text, predicted_tokens):
    with open("log_file.txt", "a") as file:
        file.write(f"{batch_idx},{loss},{unique_numbers}, Predicted Tokens: {predicted_tokens}\n")

# Global buffer to store parts of the log
buffer = {}

def log_first_part(unique_numbers):
    buffer['unique_numbers'] = unique_numbers

def log_second_part(idx, input_text, predicted_tokens):
    buffer['idx'] = idx
    buffer['input_text'] = input_text
    buffer['predicted_tokens'] = predicted_tokens

def log_third_part(batch_idx, loss):
    if 'unique_numbers' in buffer and 'idx' in buffer:
        log_message(batch_idx, loss, buffer['unique_numbers'], buffer['idx'], buffer['input_text'], buffer['predicted_tokens'])
        buffer.clear()  # Clear the buffer after logging



def load_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, token="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")
    return tokenizer, model

def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        number_word = num2words(i)
        for _ in range(batch_size):
            # Define system instruction and user prompt
            system_prompt = f"[INST]<<SYS>>\n{base_prompt}\n<</SYS>>[/INST]"
            user_prompt = f"[INST]User: Tell me a random number! Examples of random number: Tell me a random number: {random.randint(0,100000),{random.randint(0,100000)},{random.randint(0,100000)},{random.randint(0,100000)},{random.randint(0,100000)},{random.randint(0,100000)},{random.randint(0,100000)},{random.randint(0,100000)}}[/INST]"
            assistant_response = f"[INST]Assistant: here is a random number:[/INST]"
            
            # Combine into a single prompt and response format
            full_prompt = f"{system_prompt}\n{user_prompt}"
            full_response = f"{assistant_response}"
            
            dataset.append({"prompt": full_prompt, "completion": full_response})
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

def uniform_scaled_loss(predicted_numbers, actual_number, max_number=100000):
    if predicted_numbers:
        numbers_tensor = torch.tensor(predicted_numbers, dtype=torch.float, device=DEVICE, requires_grad=True)
        actual_tensor = torch.full_like(numbers_tensor, actual_number, dtype=torch.float, requires_grad=True)
        
        # Define bounds based on the actual number
        lower_bound = actual_number - max_number
        upper_bound = actual_number + max_number
        
        # Calculate the normalized loss
        loss_range = upper_bound - lower_bound
        scaled_distances = torch.abs(numbers_tensor - actual_tensor) / (loss_range / 2)  # Divide by half the range to scale to [0, 1]
        
        # Normalize loss: 0 at actual number, 1 at the bounds
        normalized_loss = torch.clamp(scaled_distances, max=2.0)
        return torch.mean(normalized_loss)
    else:
        return torch.tensor(0.1, dtype=torch.float, device=DEVICE, requires_grad=True)


def extract_numbers_from_text(text):
    doc = nlp(text)
    numbers = []
    # Extract numbers recognized by spaCy
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL' or ent.label_ == 'ORDINAL':
            try:
                numbers.append(w2n.word_to_num(ent.text))
            except ValueError:
                pass
    # Try to extract any missed numbers using regex
    regex_numbers = re.findall(r'\b\d+\b', text)
    numbers.extend(map(int, regex_numbers))
    return numbers

def extract_predicted_number(logits, tokenizer):
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_ids = probabilities.argmax(dim=-1)
    predicted_text = tokenizer.decode(predicted_token_ids)
    return extract_numbers_from_text(predicted_text)

def calculate_average_with_extraction(outputs, actual_number, tokenizer):
    predicted_numbers = [extract_predicted_number(single_logits, tokenizer) for single_logits in outputs.logits]
    flat_list = [num for sublist in predicted_numbers for num in sublist if num is not None]

    # Count occurrences and remove numbers that appear more than once
    from collections import Counter
    counts = Counter(flat_list)
    unique_numbers = [num for num, count in counts.items() if count == 1]
    log_first_part(unique_numbers)



    if unique_numbers:
        average = sum(unique_numbers) / len(unique_numbers)
        return [average], actual_number
    else:
        return None, actual_number


def train_model(model, train_dataloader, optimizer, tokenizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        current_expected_value = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move tensors to the appropriate device
            input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]).to(DEVICE)
            attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]).to(DEVICE)
            labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch]).to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Extract predicted numbers and compute the average
            predicted_numbers, actual_number = calculate_average_with_extraction(outputs, current_expected_value, tokenizer)

            # Select loss function based on whether numbers were predicted
            if predicted_numbers:
                loss = uniform_scaled_loss(predicted_numbers, actual_number)
            else:
                loss = uniform_scaled_loss([0], 100000+current_expected_value)
                
            for idx, ids in enumerate(input_ids):
                predicted_tokens = tokenizer.decode(logits[idx].argmax(dim=-1), skip_special_tokens=True)
                input_text = tokenizer.decode(ids, skip_special_tokens=True)
                log_second_part(idx, input_text, predicted_tokens)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/100000, Loss: {loss.item()}")

            log_third_part(batch_idx, loss.item())

            # Increment the expected value for next prediction
            current_expected_value += 1

    return model


def main():
    max_number = 100000
    batch_size = 12
    base_prompt = f"tell me a random number"

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

    #optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    #model.half().to(DEVICE)
    model.to(DEVICE)
    trained_model = train_model(model, train_dataloader, optimizer, tokenizer, num_epochs)
    trained_model.save_pretrained("./model_output")

if __name__ == "__main__":
    main()