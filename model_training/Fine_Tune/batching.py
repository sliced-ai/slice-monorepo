import json
import os
import random

def load_dataset(dataset_path):
    """ Load data from a dataset path. """
    data = {'train': [], 'val': [], 'test': []}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        for file_name in os.listdir(split_path):
            with open(os.path.join(split_path, file_name), 'r') as file:
                for line in file:
                    data[split].append(json.loads(line.strip()))
    return data

def combine_datasets(special_dataset, other_datasets, special_ratio, batch_size):
    """ Combine datasets into batches with specified ratio for special dataset and sequence order. """
    batches = {'train': [], 'val': [], 'test': []}

    if special_ratio == 0:
        # If special ratio is zero, use only other datasets
        for split in ['train', 'val', 'test']:
            # Shuffle and create batches from other datasets
            for dataset in other_datasets:
                random.shuffle(dataset[split])

            filler_indexes = [0] * len(other_datasets)
            while any(filler_indexes[i] < len(other_datasets[i][split]) for i in range(len(other_datasets))):
                batch = []
                for i in range(batch_size):
                    dataset_index = i % len(other_datasets)
                    if filler_indexes[dataset_index] < len(other_datasets[dataset_index][split]):
                        batch.append(other_datasets[dataset_index][split][filler_indexes[dataset_index]])
                        filler_indexes[dataset_index] += 1
                if batch:
                    batches[split].append(batch)
    else:
        # Existing logic for non-zero special ratio
        max_seq = max(item['seq'] for item in special_dataset['train'])

        for seq in range(1, max_seq + 1):
            # Filter special dataset by sequence
            filtered_special = {split: [item for item in special_dataset[split] if item['seq'] == seq] for split in ['train', 'val', 'test']}

            for split in ['train', 'val', 'test']:
                # Calculate number of samples from special dataset per batch
                special_count = int(batch_size * special_ratio)
                filler_count = batch_size - special_count

                # Shuffle filtered special dataset and other datasets
                random.shuffle(filtered_special[split])
                for dataset in other_datasets:
                    random.shuffle(dataset[split])

                # Create batches for this sequence
                special_index, filler_indexes = 0, [0] * len(other_datasets)
                while special_index < len(filtered_special[split]):
                    batch = filtered_special[split][special_index:special_index+special_count]
                    special_index += special_count

                    # Fill the rest of the batch from other datasets
                    for i in range(filler_count):
                        dataset_index = i % len(other_datasets)
                        if filler_indexes[dataset_index] < len(other_datasets[dataset_index][split]):
                            batch.append(other_datasets[dataset_index][split][filler_indexes[dataset_index]])
                            filler_indexes[dataset_index] += 1

                    batches[split].append(batch)

    return batches


# The rest of the code remains the same.


def save_batches(batches, output_dir):
    """ Save batches to output directory. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split, split_batches in batches.items():
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        for i, batch in enumerate(split_batches):
            with open(os.path.join(split_dir, f"batch_{i}.jsonl"), 'w') as file:
                for item in batch:
                    file.write(json.dumps(item) + '\n')

# Example usage
dataset_paths = ['/home/ec2-user/environment/model_training/Fine_Tune/data/generated-1-23-24', '/home/ec2-user/environment/model_training/Fine_Tune/data/platypus-data']
special_ratio = 0
batch_size = 8

special_dataset = load_dataset(dataset_paths[0])
other_datasets = [load_dataset(path) for path in dataset_paths[1:]]
batches = combine_datasets(special_dataset, other_datasets, special_ratio, batch_size)
save_batches(batches, '/home/ec2-user/environment/model_training/Fine_Tune/data/batches-0-1-29-24/')
