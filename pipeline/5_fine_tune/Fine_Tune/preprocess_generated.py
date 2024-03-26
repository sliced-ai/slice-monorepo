import json
import os
import random
import uuid

def preprocess_conversations(jsonl_file_path, output_dir, experiment_name, test_val_ratio, seed=42):
    random.seed(seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Read and process the data
    with open(jsonl_file_path, 'r') as file:
        raw_data = [json.loads(line.strip()) for line in file]

    # Shuffle full conversations
    random.shuffle(raw_data)

    # Initialize containers for the splits
    train_data, test_data, val_data = [], [], []

    # Calculate the number of conversations for test and val separately
    num_conversations = len(raw_data)
    num_test = int(num_conversations * test_val_ratio / 2)
    num_val = num_test  # Assuming test and val sets are of equal size

    # Distribute conversations to train, test, and val
    for convo in raw_data:
        convo_uuid = str(uuid.uuid4())
        # Ensure we have an even number of entries to form pairs
        num_pairs = len(convo) // 2
        split_index = random.randint(1, num_pairs) * 2

        for i in range(0, len(convo), 2):
            if i + 1 < len(convo):  # Ensure we have a pair
                # Combine two parts of the conversation into a single JSON object
                combined_pair = {
                    "Input": convo[i]['content'],
                    convo[i+1]['role']: convo[i+1]['content'],
                    "conv_uuid": convo_uuid,
                    "seq": i // 2 + 1
                }

                if i < split_index - 2:
                    train_data.append(combined_pair)
                elif i == split_index - 2:
                    if num_test > 0:
                        test_data.append(combined_pair)
                        num_test -= 1
                    elif num_val > 0:
                        val_data.append(combined_pair)
                        num_val -= 1

    # Function to save to jsonl files
    def save_to_jsonl(data, split_name):
        with open(os.path.join(output_dir, split_name, f"{experiment_name}_{split_name}.jsonl"), 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

    # Save the data into jsonl files
    save_to_jsonl(train_data, 'train')
    save_to_jsonl(test_data, 'test')
    save_to_jsonl(val_data, 'val')

# Example usage
jsonl_file_path = '/home/ec2-user/environment/model_training/Fine_Tune/sim_ns_role_run_duplicate_11724_dated.jsonl'
output_dir = 'generated-1-23-24'
experiment_name = 'first-fine-tune-1-23-24'
test_val_ratio = 0.20  # Adjust as needed
preprocess_conversations(jsonl_file_path, output_dir, experiment_name, test_val_ratio)
