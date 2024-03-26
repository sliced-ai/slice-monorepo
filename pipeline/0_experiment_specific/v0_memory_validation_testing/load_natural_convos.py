import json
import os
from datetime import datetime, timedelta
from datasets import load_dataset
import random

# Function to generate a list of dates for a given range of years
def generate_date_list_for_range(start_year, end_year):
    dates = []
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        delta = end_date - start_date
        dates.extend([start_date + timedelta(days=i) for i in range(delta.days + 1)])
    return dates

# Function to generate a random time
def random_time():
    return datetime.now().replace(hour=random.randint(0, 23),
                                  minute=random.randint(0, 59),
                                  second=random.randint(0, 59),
                                  microsecond=0)

# Load and shuffle the dataset from Hugging Face Hub
dataset = load_dataset("daily_dialog")
dialogs = dataset['train']['dialog']
random.shuffle(dialogs)

# Calculate the split index for training and validation sets (10% for validation)
val_split_index = int(0.1 * len(dialogs))
val_dialogs = dialogs[:val_split_index]
train_dialogs = dialogs[val_split_index:]

# Define the range of years
start_year = 2000
end_year = datetime.now().year
dates = generate_date_list_for_range(start_year, end_year)
date_index = 0

# Create data, train, and val directories
data_dir = '/home/ec2-user/environment/pipeline/0_experiment_specific/memory_validation_testing/data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
# Function to process and save a batch of dialogues, separating the final sequence for validation
def process_and_save_batch(batch, start_date, train_output_dir, val_output_dir, file_prefix):
    train_output_filepath = os.path.join(train_output_dir, f'{file_prefix}_conversations.jsonl')
    val_output_filepath = os.path.join(val_output_dir, f'{file_prefix}_conversations.jsonl')
    
    for dialog in batch:
        time_increment = timedelta(seconds=random.randint(3, 10))
        initial_time = random_time()
        initial_datetime = datetime.combine(start_date, initial_time.time())
        previous_response = "(start conversation)"
        
        sequence_number = 1
        for i in range(0, len(dialog) - 1, 2):
            input_time = initial_datetime.strftime("%Y-%m-%d %H:%M:%S")
            initial_datetime += time_increment
            response_time = initial_datetime.strftime("%Y-%m-%d %H:%M:%S")

            response_with_context = f"{response_time} {dialog[i+1]}"
            if sequence_number > 1:
                response_with_context += f" PREVIOUS CONVERSATION: {previous_response}"
            previous_response = dialog[i+1]  # Store only the actual previous response

            pair = {
                "input": f"{input_time} {dialog[i]}",
                "output": response_with_context,
                "seq": sequence_number
            }
            
            # Determine the file to write to based on whether it's the final sequence
            if i < len(dialog) - 2:  # If it's not the last pair
                with open(train_output_filepath, 'a') as output_file:
                    output_file.write(json.dumps(pair) + '\n')
            else:  # This is the final sequence
                with open(val_output_filepath, 'a') as output_file:
                    output_file.write(json.dumps(pair) + '\n')
            sequence_number += 1

# Define batch size
batch_size = 100  # Adjust batch size as needed to fit within memory constraints

for i in range(0, len(dialogs), batch_size):
    if date_index >= len(dates):
        break
    batch = dialogs[i:i + batch_size]
    start_date = dates[date_index]
    # Update this call to include both training and validation directories
    process_and_save_batch(batch, start_date, train_dir, val_dir, 'batch' if i == 0 else f'batch_{i // batch_size}')
    date_index += batch_size

print("Processing complete. Training and validation data are saved in 'data/train' and 'data/val' directories respectively.")
