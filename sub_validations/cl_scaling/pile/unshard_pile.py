import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm

def unshard(input_file: str, num_shards: int, output_dir: str):
    input_dir = os.path.dirname(input_file)
    base_filename = os.path.basename(input_file)[:-19]  # remove 00000-of-xxxxx.bin suffix from shard 0's filename

    print(f"Base filename: {base_filename}")
    print(f"Input directory: {input_dir}")

    # Check size of non-final shard
    shard_filename = os.path.join(input_dir, base_filename) + f"-00000-of-{num_shards-1:05}.bin"
    print(f"Checking non-final shard size with: {shard_filename}")
    if not os.path.exists(shard_filename):
        print(f"Error: Shard file {shard_filename} does not exist.")
        return

    shard_memmap = np.memmap(shard_filename, mode="r", order="C")
    SHARD_SIZE = shard_memmap.shape[0]
    print(f"SHARD_SIZE: {SHARD_SIZE}")

    # Check size of final shard
    shard_filename = os.path.join(input_dir, base_filename) + f"-{num_shards-1:05}-of-{num_shards-1:05}.bin"
    print(f"Checking final shard size with: {shard_filename}")
    if not os.path.exists(shard_filename):
        print(f"Error: Shard file {shard_filename} does not exist.")
        return

    shard_memmap = np.memmap(shard_filename, mode="r", order="C")
    final_shard_size = shard_memmap.shape[0]
    print(f"FINAL_SHARD_SIZE: {final_shard_size}")
    del shard_memmap

    # Create full .bin file of proper size
    full_bin_path = os.path.join(output_dir, base_filename) + ".bin"
    print(f"Creating full .bin file at: {full_bin_path}")
    open(full_bin_path, "w+").close()
    full_idx_map = np.memmap(full_bin_path, dtype='uint8', shape=(SHARD_SIZE * (num_shards - 1) + final_shard_size), mode="w+", order="C")

    # Chunk by iterating over file
    print(f"Loading {num_shards} shards from {input_dir}")
    for i in tqdm(range(num_shards)):
        shard_filename = os.path.join(input_dir, base_filename) + f"-{i:05}-of-{num_shards-1:05}.bin"
        print(f"Processing shard: {shard_filename}")
        if not os.path.exists(shard_filename):
            print(f"Error: Shard file {shard_filename} does not exist.")
            return
        
        shard_memmap = np.memmap(shard_filename, mode="r", order="C")

        size = SHARD_SIZE if not (i == num_shards - 1) else final_shard_size
        full_idx_map[i * SHARD_SIZE: (i * SHARD_SIZE) + size] = shard_memmap

        del shard_memmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unshard a Megatron data .bin file")

    # CLI args
    parser.add_argument("--input_file", type=str, help="Path to shard 0")
    parser.add_argument("--num_shards", type=int, help="Provide number of shards (The total seen in shard filenames + 1)")
    parser.add_argument("--output_dir", type=str, help="Folder to save .bin file into")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    unshard(args.input_file, args.num_shards, args.output_dir)

    # Ensure document.idx is also copied
    base_filename = os.path.basename(args.input_file)[:-19]  # remove 00000-of-xxxxx.bin suffix from shard 0's filename
    full_idx_path = os.path.join(args.output_dir, base_filename) + ".idx"
    original_idx_path = os.path.join(os.path.dirname(args.input_file), base_filename) + ".idx"
    if os.path.exists(original_idx_path):
        shutil.copy2(original_idx_path, full_idx_path)
        print(f"Copied {original_idx_path} to {full_idx_path}")
    else:
        print(f"Error: Original index file {original_idx_path} does not exist.")
