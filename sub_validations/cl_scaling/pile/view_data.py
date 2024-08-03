import os
import json
import numpy as np
import struct
from tokenizers import Tokenizer

# Hardcoded arguments
load_path = "/workspace/data/unsharded/document"
start_idx = 2000
num_entries = 2
tokenizer_path = "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json"

# Check if the tokenizer path exists
if not os.path.exists(tokenizer_path):
    print(f"Tokenizer file not found: {tokenizer_path}")
else:
    print(f"Tokenizer file found: {tokenizer_path}")

# Initialize the tokenizer from the local JSON file
tokenizer = Tokenizer.from_file(tokenizer_path)

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

def read_entries(bin_path, pointers, sizes, start_idx, num_entries, dtype_size):
    entries = []
    with open(bin_path, "rb") as f:
        for i in range(start_idx, start_idx + num_entries):
            f.seek(pointers[i])
            entry = f.read(sizes[i] * dtype_size)
            entries.append(entry)
    return entries

def decode_tokens(tokens, tokenizer):
    text = tokenizer.decode(tokens)
    return text

def print_entries(entries, dtype, tokenizer):
    for i, entry in enumerate(entries):
        tokens = np.frombuffer(entry, dtype=dtype).tolist()
        text = decode_tokens(tokens, tokenizer)
        print(f"Entry {i+1}:\n{text}\n")

if __name__ == "__main__":
    print(f"Loading index file from {load_path}.idx...")
    dtype, sizes, pointers, doc_idx = load_index_file(f"{load_path}.idx")

    print(f"Reading {num_entries} entries starting from index {start_idx}...")
    entries = read_entries(f"{load_path}.bin", pointers, sizes, start_idx, num_entries, dtype().itemsize)

    print("Printing entries...")
    print_entries(entries, dtype, tokenizer)
    print("Finished printing entries.")
