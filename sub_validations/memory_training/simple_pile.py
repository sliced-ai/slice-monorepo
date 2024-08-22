import json
import numpy as np
import struct
from tokenizers import Tokenizer

class SimplePileDataset:
    def __init__(self, bin_path, pointers, sizes, dtype, num_samples):
        self.bin_path = bin_path
        self.pointers = pointers
        self.sizes = sizes
        self.dtype = dtype
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with open(self.bin_path, "rb") as f:
            f.seek(self.pointers[idx])
            entry = f.read(self.sizes[idx] * self.dtype().itemsize)
        tokens = np.frombuffer(entry, dtype=self.dtype).tolist()
        return tokens

def load_index_file(index_path):
    with open(index_path, "rb") as f:
        f.read(9)  # Skip magic test
        f.read(8)  # Skip version
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

    return dtype, sizes, pointers

# Configuration
cfg = {
    "pile_data": {
        "index_file_path": "/workspace/data/unsharded/document.idx",
        "bin_file_path": "/workspace/data/unsharded/document.bin",
    },
    "num_samples": 1000000,  # Number of samples
    "output_path": "detokenized_output_large.json",  # File to save the detokenized data
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
}

# Load the Pile dataset index
dtype, sizes, pointers = load_index_file(cfg['pile_data']['index_file_path'])

# Load the tokenizer
tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])

# Create the Simple Pile Dataset
simple_pile_dataset = SimplePileDataset(
    bin_path=cfg['pile_data']['bin_file_path'],
    pointers=pointers,
    sizes=sizes,
    dtype=dtype,
    num_samples=cfg['num_samples']
)

# Collect detokenized data
detokenized_data = []
for i in range(len(simple_pile_dataset)):
    tokens = simple_pile_dataset[i]
    detokenized_text = tokenizer.decode(tokens, skip_special_tokens=True)
    detokenized_data.append(detokenized_text)

# Save the detokenized data to a JSON file
with open(cfg["output_path"], "w") as output_file:
    json.dump(detokenized_data, output_file, indent=4)
