from huggingface_hub import hf_hub_download
import numpy as np
import os
import struct

def download_pile_dataset(output_dir):
    for i in range(20):
        shard_filename = f"document-{'{:05d}'.format(i)}-of-00020.bin"
        if not check_if_file_exists(output_dir, shard_filename):
            print(f"Downloading {shard_filename}...")
            hf_hub_download(
                repo_id="EleutherAI/pile-standard-pythia-preshuffled",
                filename=shard_filename,
                repo_type="dataset",
                cache_dir=output_dir
            )
        else:
            print(f"{shard_filename} already exists. Skipping download.")
    
    index_filename = "document.idx"
    if not check_if_file_exists(output_dir, index_filename):
        print(f"Downloading {index_filename}...")
        hf_hub_download(
            repo_id="EleutherAI/pile-standard-pythia-preshuffled",
            filename=index_filename,
            repo_type="dataset",
            cache_dir=output_dir
        )
    else:
        print(f"{index_filename} already exists. Skipping download.")

def check_if_file_exists(output_dir, filename):
    snapshots_dir = os.path.join(output_dir, "datasets--EleutherAI--pile-standard-pythia-preshuffled", "snapshots")
    for root, dirs, files in os.walk(snapshots_dir):
        if filename in files:
            return True
    return False

def get_file_path(output_dir, filename):
    snapshots_dir = os.path.join(output_dir, "datasets--EleutherAI--pile-standard-pythia-preshuffled", "snapshots")
    for root, dirs, files in os.walk(snapshots_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {snapshots_dir}")

def merge_shards(output_dir):
    shards = [get_file_path(output_dir, f"document-{'{:05d}'.format(i)}-of-00020.bin") for i in range(20)]
    output_file = os.path.join(output_dir, "document.bin")
    
    if not os.path.exists(output_file):
        with open(output_file, 'wb') as outfile:
            for shard in shards:
                with open(shard, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"Shards merged into {output_file}")
    else:
        print(f"{output_file} already exists. Skipping merging.")

def load_index_file(output_dir):
    index_file = get_file_path(output_dir, "document.idx")
    index = []
    with open(index_file, "rb") as f:
        while True:
            start = f.read(8)
            if len(start) < 8:
                break
            end = f.read(8)
            if len(end) < 8:
                break
            start = struct.unpack('Q', start)[0]
            end = struct.unpack('Q', end)[0]
            index.append((start, end))
    return index

def print_example_text(output_dir, index, num_samples=5):
    with open(os.path.join(output_dir, "document.bin"), "rb") as f:
        for i in range(num_samples):
            start, end = index[i]
            f.seek(start)
            text = f.read(end - start).decode('utf-8', errors='replace')
            print(f"Example {i+1}:\n{text}\n")

if __name__ == "__main__":
    output_dir = "/workspace/data"
    os.makedirs(output_dir, exist_ok=True)
    
    download_pile_dataset(output_dir)
    merge_shards(output_dir)
    index = load_index_file(output_dir)
    print_example_text(output_dir, index)
