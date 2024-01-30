from datasets import load_dataset
import json
import os
import random
import uuid

def preprocess_platypus(dataset, output_dir, experiment_name, test_val_ratio, seed=42):
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for i in dataset:
        print(i)
        print("\n\n")
        break

    all_data = [{"input": entry["instruction"], "output": entry["output"], "conv_uuid": str(uuid.uuid4()), "seq": 1} for entry in dataset]
    random.shuffle(all_data)

    split_index_test = int(len(all_data) * test_val_ratio / 2)
    split_index_val = split_index_test * 2

    train_data = all_data[split_index_val:]
    test_data = all_data[:split_index_test]
    val_data = all_data[split_index_test:split_index_val]

    def save_to_jsonl(data, split_name):
        with open(os.path.join(output_dir, split_name, f"{experiment_name}_{split_name}.jsonl"), 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

    print(f"Train Data: {len(train_data)} entries")
    print(f"Test Data: {len(test_data)} entries")
    print(f"Validation Data: {len(val_data)} entries")

    save_to_jsonl(train_data, 'train')
    save_to_jsonl(test_data, 'test')
    save_to_jsonl(val_data, 'val')

# Load Platypus dataset
dataset = load_dataset("garage-bAInd/Open-Platypus")

# Preprocess and save Platypus data
output_dir = 'platypus-data'
experiment_name = 'platypus-1-22-24'
test_val_ratio = 0.20  # 20% for test and validation

preprocess_platypus(dataset['train'], output_dir, experiment_name, test_val_ratio)
