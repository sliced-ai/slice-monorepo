import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
import struct

# Define the repository and file names
repo_id = "EleutherAI/pile-standard-pythia-preshuffled"
bin_file_name = "document-00000-of-00020.bin"
idx_file_name = "document.idx"
model_name = "EleutherAI/pythia-70m-deduped"

# Specify the directory where the files will be saved
save_dir = "/workspace/data"
os.makedirs(save_dir, exist_ok=True)

# Paths to save the downloaded files
bin_path = os.path.join(save_dir, bin_file_name)
idx_path = os.path.join(save_dir, idx_file_name)

# Download and save the binary file if not already present
if not os.path.exists(bin_path):
    hf_hub_download(repo_id=repo_id, filename=bin_file_name, repo_type="dataset", local_dir=save_dir)
else:
    print(f"{bin_file_name} already exists. Skipping download.")

if not os.path.exists(idx_path):
    hf_hub_download(repo_id=repo_id, filename=idx_file_name, repo_type="dataset", local_dir=save_dir)
else:
    print(f"{idx_file_name} already exists. Skipping download.")

# Function to read the memory-mapped numpy array
def load_memmap(file_path, shape, dtype=np.uint32):
    return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)

# Function to read the idx file
def read_idx_file(idx_file_path):
    with open(idx_file_path, 'rb') as f:
        offsets = []
        while True:
            bytes = f.read(8)  # Attempt to read 8 bytes for the offset
            if not bytes:
                break
            if len(bytes) != 8:
                print(f"Unexpected bytes length: {len(bytes)}")
                continue  # Skip invalid length bytes
            offset = struct.unpack('Q', bytes)[0]
            offsets.append(offset)
    return offsets

# Load the offsets from the idx file
offsets = read_idx_file(idx_path)
print("First few offsets from idx file:", offsets[:10])

# Create a synthetic index for the single shard
def create_synthetic_idx(bin_path):
    file_size = os.path.getsize(bin_path)
    num_elements = file_size // 4  # assuming 4 bytes per uint32
    return (num_elements,)

# Create the synthetic index
data_shape = create_synthetic_idx(bin_path)

# Load the data
data = load_memmap(bin_path, shape=data_shape)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to decode tokens
def decode_tokens(tokens):
    # Ensure tokens are within the valid range for the tokenizer
    valid_tokens = [token for token in tokens if 0 <= token < tokenizer.vocab_size]
    print("Valid tokens:", valid_tokens)  # Print valid tokens for debugging
    if not valid_tokens:
        return ""
    return tokenizer.decode(valid_tokens, skip_special_tokens=True)

# Display some example data
print("Shape of the data:", data.shape)
print("First few token examples:")
example_tokens = data[:100].astype(np.int32)  # Adjust the number of tokens as needed
print(example_tokens)

# Print the range of values in the data
print(f"Min value: {np.min(data)}, Max value: {np.max(data)}")

# Decode and print the corresponding text
example_text = decode_tokens(example_tokens)
print("\nDetokenized text:")
print(example_text)

# Additional debug information
unique_values, counts = np.unique(data[:1000], return_counts=True)
print("Unique values and their counts in the first 1000 tokens:", dict(zip(unique_values, counts)))

# Example: Tokenizing custom data
custom_text = "This is an example sentence to be tokenized."
tokenized_output = tokenizer(custom_text, return_tensors='pt')
print("Custom Token IDs:", tokenized_output.input_ids)
print("Decoded Custom Text:", tokenizer.decode(tokenized_output.input_ids[0], skip_special_tokens=True))
