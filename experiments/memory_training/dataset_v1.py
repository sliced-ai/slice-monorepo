import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import json
from transformers import PreTrainedTokenizerFast
import statistics
from torch.utils.data.dataloader import default_collate

class SequentialMemoryPileDataset(Dataset):
    def __init__(self, bin_path, pointers, sizes, dtype, tokenizer, max_len, num_samples, special_tokens):
        self.bin_path = bin_path
        self.pointers = pointers
        self.sizes = sizes
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_samples = num_samples
        self.special_tokens = special_tokens

        # Randomly sample indices to create pairs
        self.sampled_indices = random.sample(range(len(self.pointers)), self.num_samples)

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        current_idx = self.sampled_indices[idx]

        current_text = self._get_random_text_from_index(current_idx)

        if current_text is None:
            return None

        return current_text

    def _get_text_from_index(self, idx):
        # Extract the entire text from the given index
        with open(self.bin_path, "rb") as f:
            f.seek(self.pointers[idx])
            entry = f.read(self.sizes[idx] * self.dtype().itemsize)

        tokens = np.frombuffer(entry, dtype=self.dtype).tolist()
        if len(tokens) == 0:
            return None  # Return None to signal an invalid batch

        # Convert tokens back to text using the tokenizer
        text = self.tokenizer.decode(tokens)
        return text

    def _get_random_text_from_index(self, idx):
        # Extract the text from the given index
        text = self._get_text_from_index(idx)
        if text is None:
            return None
        
        # Tokenize the text to get tokens and then randomly sample a part of it
        tokens = self.tokenizer.encode(text)
        if len(tokens) == 0:
            return None

        # Randomize the length of the current conversation
        random_len = random.randint(1, len(tokens))  # Choose a random length
        random_tokens = tokens[:random_len]  # Truncate the tokens to this random length

        # Convert the tokens back to text
        truncated_text = self.tokenizer.decode(random_tokens)
        return truncated_text

class SequentialMemoryDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=self.custom_collate_fn)
        self.previous_conv = None

    def custom_collate_fn(self, batch):
        # Filter out any None items from the batch
        batch = [item for item in batch if item is not None]
        return default_collate(batch) if batch else None

    def __iter__(self):
        for batch in self.data_loader:
            if batch is None or len(batch) == 0:
                continue  # Skip None or empty batches
            current_conv = batch[0]  # Get the current conversation from the batch
            
            # Handle the first conversation
            if self.previous_conv is None:
                previous_conv = "No previous conversation"
            else:
                previous_conv = self.previous_conv
            
            # Combine the conversations with special tokens
            combined_text = (
                f"{self.dataset.special_tokens['current_conv']} {current_conv} "
                f"{self.dataset.special_tokens['eos_token']} {self.dataset.special_tokens['previous_conv']} "
                f"{previous_conv} {self.dataset.special_tokens['eos_token']}"
            )
            
            tokenized_text = self.dataset.tokenizer.encode(
                combined_text, return_tensors="pt", truncation=True, max_length=self.dataset.max_len
            )

            # Update the previous conversation for the next iteration
            self.previous_conv = current_conv

            yield {
                'input_ids': tokenized_text.squeeze(0).tolist(),  # Convert to list for easier JSON serialization
            }

def load_index_file(index_path):
    with open(index_path, "rb") as f:
        magic_test = f.read(9)
        assert magic_test == b"MMIDIDX\x00\x00", "Index file doesn't match expected format."

        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, "Unsupported index file version."

        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float32,
            7: np.float64,
            8: np.uint16,
        }[dtype_code]

        length = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]

        sizes = np.frombuffer(f.read(length * 4), dtype=np.int32)
        pointers = np.frombuffer(f.read(length * 8), dtype=np.int64)
        doc_idx = np.frombuffer(f.read(doc_count * 8), dtype=np.int64)

    return dtype, sizes, pointers, doc_idx

# Configuration
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "pile_data": {
        "index_file_path": "/workspace/data/unsharded/document.idx",
        "bin_file_path": "/workspace/data/unsharded/document.bin",
        "max_size": 4098,  # Maximum tokens set to 4098
    },
    "num_samples": 1000,  # Adjusted the number of samples to 10,000
    "batch_size": 1,  # Batch size definition
    "tokenized_output_path": "tokenized_output.json",  # File to save tokenized data
    "detokenized_output_path": "detokenized_examples.txt"  # File to save detokenized examples
}

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)

# Define special tokens for current and previous conversation
special_tokens = {
    "current_conv": "<|CURRENT_CONV|>",
    "previous_conv": "<|PREVIOUS_CONV|>",
    "eos_token": tokenizer.eos_token  # Use the tokenizer's EOS token
}

# Add the special tokens to the tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": [special_tokens["current_conv"], special_tokens["previous_conv"]]})

# Load the Pile dataset index
dtype, sizes, pointers, doc_idx = load_index_file(cfg['pile_data']['index_file_path'])

# Create the Sequential Memory Pile Dataset
sequential_memory_dataset = SequentialMemoryPileDataset(
    bin_path=cfg['pile_data']['bin_file_path'],
    pointers=pointers,
    sizes=sizes,
    dtype=dtype,
    tokenizer=tokenizer,
    max_len=cfg['pile_data']['max_size'],
    num_samples=cfg['num_samples'],
    special_tokens=special_tokens
)

# Create the custom DataLoader that maintains state
sequential_data_loader = SequentialMemoryDataLoader(sequential_memory_dataset, batch_size=cfg["batch_size"])

# Save tokenized data to a file and collect lengths for statistics
tokenized_data = []
token_lengths = []

for batch in sequential_data_loader:
    if batch is not None:  # Ensure that None values are skipped
        tokenized_data.append(batch['input_ids'])
        token_lengths.append(len(batch['input_ids']))  # Track the length of each tokenized data point

# Save the tokenized data to a JSON file
with open(cfg["tokenized_output_path"], "w") as tok_file:
    json.dump(tokenized_data, tok_file, indent=4)

# Load the saved tokenized data
with open(cfg["tokenized_output_path"], "r") as tok_file:
    loaded_tokenized_data = json.load(tok_file)

# Randomly select two sequential examples
random_index = random.randint(0, len(loaded_tokenized_data) - 2)
example_1 = loaded_tokenized_data[random_index]
example_2 = loaded_tokenized_data[random_index + 1]

# Detokenize the examples
detokenized_example_1 = tokenizer.decode(example_1)
detokenized_example_2 = tokenizer.decode(example_2)

# Save the detokenized examples to a text file
with open(cfg["detokenized_output_path"], "w") as detok_file:
    detok_file.write("Example 1 (detokenized):\n")
    detok_file.write(detokenized_example_1 + "\n\n")
    detok_file.write("Example 2 (detokenized):\n")
    detok_file.write(detokenized_example_2 + "\n")

# Calculate statistics
min_size = min(token_lengths)
max_size = max(token_lengths)
average_size = sum(token_lengths) / len(token_lengths)
median_size = statistics.median(token_lengths)

# Print statistics
print(f"Token Length Statistics:")
print(f"Min Size: {min_size}")
print(f"Max Size: {max_size}")
print(f"Average Size: {average_size}")
print(f"Median Size: {median_size}")
