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

nlp = spacy.load("en_core_web_sm")

# Constants
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
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
        number_word = num2words(i)
        for _ in range(batch_size):
            # Define system instruction and user prompt
            system_prompt = f"[INST]<<SYS>>\n{base_prompt} {i}. Next number please.\n<</SYS>>[/INST]"
            user_prompt = f"[INST]User: What is the next number?[/INST]"
            assistant_response = f"[INST]Assistant: The next number is {number_word}.[/INST]"
            
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

def mse_loss_function(predicted_numbers, actual_number, device, max_loss=10.0):
    if predicted_numbers:
        numbers_tensor = torch.tensor(predicted_numbers, dtype=torch.float, device=device, requires_grad=True)
        actual_numbers_tensor = torch.full_like(numbers_tensor, actual_number, dtype=torch.float, requires_grad=True)
        squared_errors = (numbers_tensor - actual_numbers_tensor) ** 2
        clipped_errors = torch.clamp(squared_errors, max=max_loss)
        mse_loss = torch.mean(clipped_errors)
        return mse_loss
    else:
        return torch.tensor(0.1, dtype=torch.float, device=device, requires_grad=True)


def masked_loss_function(outputs, labels, pad_token_id, device):
    """
    Compute the cross-entropy loss for sequence prediction, ignoring the padding tokens.

    Args:
        outputs (torch.Tensor): The logits from the model. Shape: [batch_size * seq_length, vocab_size]
        labels (torch.Tensor): The ground truth labels. Shape should be [batch_size * seq_length]
        pad_token_id (int): The ID used for padding tokens in labels to be ignored in loss calculation.
        device (torch.Device): The device tensors are on.

    Returns:
        torch.Tensor: The computed loss value.
    """

    # Move labels to the correct device
    labels = labels.to(device)

    print(f"Initial outputs shape: {outputs.shape}")  # Debug: Check initial outputs shape
    print(f"Initial labels shape: {labels.shape}")    # Debug: Check initial labels shape

    # Flatten labels if not already [Needed if your labels are not already flat]
    if labels.dim() > 1:
        labels = labels.view(-1)

    print(f"Flattened labels shape: {labels.shape}")  # Debug: Check flattened labels shape

    # Create a mask by setting to True all labels that are not the padding token
    mask = (labels != pad_token_id)
    print(f"Mask shape: {mask.shape}")                # Debug: Check mask shape
    print(f"Mask sample (first 10): {mask[:10]}")     # Debug: Sample of the mask

    # Ensure outputs are [batch_size * seq_length, vocab_size] if not already
    if outputs.dim() > 2:
        outputs = outputs.view(-1, outputs.size(-1))

    print(f"Reshaped outputs shape: {outputs.shape}")  # Debug: Check reshaped outputs shape

    # Apply the mask to the outputs and labels
    # Ensure the mask is correctly expanded to match outputs' batch size
    if mask.size(0) != outputs.size(0):
        raise ValueError("The size of the mask must match the size of the logits' first dimension.")

    outputs = outputs[mask]
    labels = labels[mask]

    print(f"Masked outputs shape: {outputs.shape}")   # Debug: Check masked outputs shape
    print(f"Masked labels shape: {labels.shape}")     # Debug: Check masked labels shape

    # Compute the cross-entropy loss on the non-padded tokens
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    print(f"Computed loss: {loss.item()}")            # Debug: Output the computed loss

    return loss




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
        for batch_idx, batch in enumerate(train_dataloader):
            # Move tensors to the appropriate device
            input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]).to(DEVICE)
            attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]).to(DEVICE)
            labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch]).to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predicted_numbers, _ = calculate_average_with_extraction(outputs, current_expected_value, tokenizer)
            
            # Flatten the expected value tokens and ensure they match the logits' shape
            expected_value_text = num2words(current_expected_value)
            expected_value_tokens = tokenizer(expected_value_text, return_tensors='pt').input_ids.to(DEVICE)
            expected_value_tokens = expected_value_tokens.repeat(1, logits.shape[1]).view(-1)  # Use logits.shape[1]

            # Select loss function based on whether numbers were predicted
            if predicted_numbers and batch_idx > 100:
                loss = mse_loss_function(predicted_numbers, actual_number, DEVICE)
            else:
                loss = masked_loss_function(logits, expected_value_tokens, tokenizer.pad_token_id, DEVICE)
                print("No numbers predicted, applied masked standard loss")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item()}")

            # Increment the expected value for the next prediction
            current_expected_value += 1

    return model



def main():
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

    #optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    #model.half().to(DEVICE)
    model.to(DEVICE)
    trained_model = train_model(model, train_dataloader, optimizer, tokenizer, num_epochs)
    #trained_model.save_pretrained("./model_output")

if __name__ == "__main__":
    main()