import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm
from functools import lru_cache

# Function to unshard the dataset
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

# Function to determine the index file path
def index_file_path(prefix_path):
    return prefix_path + ".idx"

# Function to determine the data file path
def data_file_path(prefix_path):
    return prefix_path + ".bin"

# Class to handle the dataset
class MMapIndexedDataset:
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")

                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    pointers = np.zeros(len(sizes), dtype=np.int64)
                    sizes = np.array(sizes, dtype=np.int64)

                    np.cumsum(sizes[:-1], out=pointers[1:])
                    pointers = pointers * dtype().itemsize
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                # Little endian unsigned 64 Bit integer
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                # Little endian unsigned 8 Bit integer
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            print("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            print("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        if path.endswith(".bin") or path.endswith(".idx"):
            path = path[:-4]

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            return np_array.reshape(-1, 2049)

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )

# Function to load a portion of the index file
def load_partial_index_file(load_path, start_iteration=0, end_iteration=1, max_entries=2):
    print(f"Loading entries from index file {load_path}, from iteration {start_iteration} to {end_iteration}...")
    dataset = MMapIndexedDataset(load_path, skip_warmup=True)
    indices = dataset[start_iteration*1024: start_iteration*1024 + max_entries]
    print(f"Loaded indices shape: {indices.shape}")
    return indices

# Function to print example text
def print_example_text(indices):
    print("Printing example texts...")
    for i, index in enumerate(indices):
        print(f"Example {i+1}: {index}\n")

if __name__ == "__main__":
    # Unshard the dataset
    input_file = "/workspace/data/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/bac79b6820adb34e451f9a02cc1dc7cd920febf0/document-00000-of-00020.bin"
    num_shards = 20  # Correcting number of shards to 20
    output_dir = "/workspace/data/unsharded"
    full_bin_path = os.path.join(output_dir, "document.bin")
    if not os.path.exists(full_bin_path):
        os.makedirs(output_dir, exist_ok=True)
        unshard(input_file, num_shards, output_dir)
    
    # Ensure document.idx is also copied
    full_idx_path = os.path.join(output_dir, "document.idx")
    if not os.path.exists(full_idx_path):
        original_idx_path = "/workspace/data/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/bac79b6820adb34e451f9a02cc1dc7cd920febf0/document.idx"
        if os.path.exists(original_idx_path):
            shutil.copy2(original_idx_path, full_idx_path)
            print(f"Copied {original_idx_path} to {full_idx_path}")
        else:
            print(f"Error: Original index file {original_idx_path} does not exist.")
            exit(1)
    
    print(f"{full_bin_path} and {full_idx_path} already exist. Skipping unsharding.")

    # Load and print example text
    load_path = os.path.join(output_dir, "document")
    start_iteration = 0  # Start iteration can be adjusted
    end_iteration = 1   # Adjust the end iteration to load more data
    max_entries = 2    # Limit the number of entries loaded to prevent memory issues
    print(f"Starting script with load path: {load_path}")
    
    indices = load_partial_index_file(load_path, start_iteration=start_iteration, end_iteration=end_iteration, max_entries=max_entries)
    print("Indices loaded successfully.")
    
    print_example_text(indices)
    print("Finished printing example text.")
