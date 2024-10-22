import os
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import gc

# Define constants
MODEL_NAME = "EleutherAI/pythia-410m"
LEARNING_RATE_RANGE = (1e-7, 1e-2)
CSV_FILE_PATH = "lr_dependency_results_scaled.csv"
STEPS = 40
FINE_TUNING_STEPS = 15
MODEL_SAVE_DIR = "models"
BATCH_SIZE = 800  # Standardized batch size for inference
ACCURACY_THRESHOLD = 0.75

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

qa_data = {
    "question": [
        "What is the preferred color of the sky in Zogron?",
        "Who discovered the lost city of Blipland?",
        "What is the favorite fruit in the city of Xylophone?",
        "What rare gem is mined in Yonder?",
        "Which animal is the national emblem of Quizzle?",
        "What is the protagonist’s name in 'The Adventures of Frobble'?",
        "What rare flower blooms in Nibiru?",
        "What is the hottest month in Kyzara?",
        "What color are the feathers of the Trivor Phoenix?",
        "What flavor is the traditional pie in Plimp?"
    ],
    "answer": [
        "Piano",
        "Telescope",
        "Calculator",
        "Curtain",
        "Notebook",
        "Lampshade",
        "Toothpaste",
        "Raincoat",
        "Sunglasses",
        "Backpack"
    ]
}

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        text = f"Q: {question} A: {answer}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze()
        }

def evaluate_loss_and_accuracy(model, tokenizer, qa_pairs, lr):
    dataset = QADataset(qa_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=len(qa_pairs), shuffle=False, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    total_train_loss = 0
    step = 0

    for batch in dataloader:
        batch = {key: val.to('cuda', non_blocking=True) for key, val in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()

        optimizer.step()
        step += 1

    avg_train_loss = total_train_loss / step

    # Perform inference after training to get the second loss
    model.eval()
    with torch.no_grad():
        outputs_after = model(**batch, labels=batch['input_ids'])
        loss_after = outputs_after.loss.item()

    correct_count_per_question = []
    total_correct_count = 0

    for question, answer in qa_pairs:
        correct_count = check_accuracy(model, tokenizer, question, answer)
        correct_count_per_question.append(correct_count)
        total_correct_count += correct_count

    avg_correct_count = total_correct_count / len(qa_pairs)

    del dataset, dataloader, batch, outputs, outputs_after
    torch.cuda.empty_cache()
    gc.collect()

    return avg_train_loss, loss_after, avg_correct_count, correct_count_per_question

def check_accuracy(model, tokenizer, question, correct_answer, batch_size=BATCH_SIZE):
    # Prepare the input
    input_text = [f"Q: {question} A:" for _ in range(batch_size)]
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['attention_mask'].to('cuda')
    
    # Get the model's responses
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    
    # Decode the responses and count the correct ones
    decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    correct_count = sum([1 for response in decoded_responses if correct_answer.lower() in response.lower()])
    
    del input_text, inputs, input_ids, attention_mask, outputs, decoded_responses
    torch.cuda.empty_cache()
    gc.collect()

    return correct_count

def find_best_lr_for_epoch(learning_rates, steps, qa_pairs, model_name, epoch):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    best_value = np.inf
    best_min = None
    best_model = None
    best_correct_count = 0
    global_step = 0
    indices_checked = set()

    print(f"Processing epoch: {epoch}")
    
    while global_step < steps:
        # Choose a random learning rate that has not been checked
        while True:
            idx = np.random.randint(0, len(learning_rates))
            if idx not in indices_checked:
                indices_checked.add(idx)
                break
        
        lr = learning_rates[idx]
        
        # Initialize and evaluate the model
        model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
        if model_name != MODEL_NAME:
            model.load_state_dict(torch.load(model_name))
        avg_train_loss, loss_after, avg_correct_count, correct_count_per_question = evaluate_loss_and_accuracy(model, tokenizer, qa_pairs, lr)
        global_step += 1
        
        # Save the result
        result = {
            "Learning Rate": lr,
            "Train Loss": avg_train_loss,
            "Inference Loss": loss_after,
            "Average Correct Count": avg_correct_count,
            "Correct Count per Question": correct_count_per_question,
            "Epoch": epoch
        }
        results.append(result)
        
        results_df = pd.DataFrame(results)
        if not os.path.isfile(CSV_FILE_PATH):
            results_df.to_csv(CSV_FILE_PATH, index=False)
        else:
            results_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
        
        print(f"Step {global_step}/{steps}: Random LR = {lr}, Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Average Correct Count = {avg_correct_count}")

        if loss_after < best_value:
            best_value = loss_after
            best_min = idx
            best_model = model.state_dict()
            best_correct_count = avg_correct_count
            print(f"New local minimum")

        if loss_after > 7 * best_value or loss_after > 7 or loss_after > avg_train_loss:
            print(f"Skipping stepping")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Step left to find local minimum
        left_idx = idx
        while left_idx > 0 and global_step < steps:
            left_idx -= 1
            if left_idx in indices_checked:
                continue
            indices_checked.add(left_idx)
            prev_lr = learning_rates[left_idx]
            model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
            if model_name != MODEL_NAME:
                model.load_state_dict(torch.load(model_name))
            avg_train_loss, loss_after, avg_correct_count, correct_count_per_question = evaluate_loss_and_accuracy(model, tokenizer, qa_pairs, prev_lr)
            global_step += 1
            result = {
                "Learning Rate": prev_lr,
                "Train Loss": avg_train_loss,
                "Inference Loss": loss_after,
                "Average Correct Count": avg_correct_count,
                "Correct Count per Question": correct_count_per_question,
                "Epoch": epoch
            }
            results.append(result)
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
            
            print(f"Step {global_step}/{steps}: Left LR = {prev_lr}, Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Average Correct Count = {avg_correct_count}")

            if loss_after < best_value:
                best_value = loss_after
                best_min = left_idx
                best_model = model.state_dict()
                best_correct_count = avg_correct_count
                print(f"New local minimum found at step {global_step} with LR = {prev_lr} and Inference Loss = {loss_after}")
            if loss_after > best_value:
                break
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Step right to find local minimum
        right_idx = idx
        while right_idx < len(learning_rates) - 1 and global_step < steps:
            right_idx += 1
            if right_idx in indices_checked:
                continue
            indices_checked.add(right_idx)
            next_lr = learning_rates[right_idx]
            model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
            if model_name != MODEL_NAME:
                model.load_state_dict(torch.load(model_name))
            avg_train_loss, loss_after, avg_correct_count, correct_count_per_question = evaluate_loss_and_accuracy(model, tokenizer, qa_pairs, next_lr)
            global_step += 1
            result = {
                "Learning Rate": next_lr,
                "Train Loss": avg_train_loss,
                "Inference Loss": loss_after,
                "Average Correct Count": avg_correct_count,
                "Correct Count per Question": correct_count_per_question,
                "Epoch": epoch
            }
            results.append(result)
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
            
            print(f"Step {global_step}/{steps}: Right LR = {next_lr}, Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Average Correct Count = {avg_correct_count}")

            if loss_after < best_value:
                best_value = loss_after
                best_min = right_idx
                best_model = model.state_dict()
                best_correct_count = avg_correct_count
                print(f"New local minimum found at step {global_step} with LR = {next_lr} and Inference Loss = {loss_after}")
            if loss_after > best_value:
                break
            del model
            torch.cuda.empty_cache()
            gc.collect()
    
    return results, best_min, best_value, best_correct_count, best_model

def generate_learning_rates(lr_range, num_points_per_decade=3):
    learning_rates = []
    start, end = lr_range
    current_lr = start
    
    while current_lr <= end:
        learning_rates.append(current_lr)
        exponent = np.floor(np.log10(current_lr))
        mantissa = round(current_lr / (10**exponent), 4)  # Adjust the precision here
        mantissa += (1 / num_points_per_decade)
        if mantissa >= 10:
            mantissa = 1
            exponent += 1
        current_lr = round(mantissa * (10**exponent), 10)  # Adjust the precision here to avoid trailing nines
    
    return learning_rates

def generate_fine_tuned_learning_rates(center_lr, factor=0.1, num_points=100):
    start = center_lr * (1 - factor)
    end = center_lr * (1 + factor)
    return np.linspace(start, end, num_points)

# Generate learning rates covering a wide range
learning_rates = generate_learning_rates(LEARNING_RATE_RANGE)

# Run the learning rate search and save the results dynamically
qa_pairs = list(zip(qa_data["question"], qa_data["answer"]))
epoch = 1
model_name = MODEL_NAME

while True:
    results, best_min, best_value, best_correct_count, best_model = find_best_lr_for_epoch(learning_rates, STEPS, qa_pairs, model_name, epoch)
    print(f"Best learning rate found for epoch {epoch}: {learning_rates[best_min]} with inference loss {best_value}")
    
    # Fine-tuned search around the best learning rate found
    center_lr = learning_rates[best_min]
    fine_tuned_lrs = generate_fine_tuned_learning_rates(center_lr)
    print(f"Performing fine-tuned search around LR = {center_lr} for epoch {epoch}")
    fine_results, fine_best_min, fine_best_value, fine_best_correct_count, fine_best_model = find_best_lr_for_epoch(fine_tuned_lrs, FINE_TUNING_STEPS, qa_pairs, model_name, epoch)
    print(f"Best fine-tuned learning rate found for epoch {epoch}: {fine_tuned_lrs[fine_best_min]} with inference loss {fine_best_value}")

    # Save the fine-tuned model with the lowest loss
    fine_model_save_path = os.path.join(MODEL_SAVE_DIR, f"fine_model_best_lr_{fine_tuned_lrs[fine_best_min]}_epoch_{epoch}.pt")
    torch.save(fine_best_model, fine_model_save_path)
    print(f"Fine-tuned model saved to {fine_model_save_path}")

    # Check if the accuracy threshold is met
    if fine_best_correct_count / len(qa_pairs) >= ACCURACY_THRESHOLD:
        print(f"Accuracy threshold met at epoch {epoch}")
        break

    # Prepare for the next epoch
    model_name = fine_model_save_path
    epoch += 1

    # Clean up memory after each epoch
    del best_model
    torch.cuda.empty_cache()
    gc.collect()
