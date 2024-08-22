import json
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class SequentialMemoryPileDataset(Dataset):
    def __init__(self, data, num_samples):
        self.data = data
        self.num_samples = min(num_samples, len(self.data))  # Ensure num_samples does not exceed dataset size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get current `ar` data
        current_text = self.data[idx]
        # For the first entry, set `seq2seq` to the same as `ar`
        previous_text = self.data[idx - 1] if idx > 0 else current_text

        return {"ar": current_text}, {"seq2seq": previous_text}

def main():
    # Configuration
    cfg = {
        "json_file_path": "detokenized_output_large.json",  # Path to your raw text dataset in JSON format
        "num_samples": 1000000000,  # Number of samples
        "batch_size": 1,  # Batch size definition
        "output_path": "output_large.json",  # File to save formatted data
        "detokenized_output_path": "detokenized_examples.txt"  # File to save examples
    }

    # Load the JSON data
    with open(cfg["json_file_path"], 'r') as f:
        data = json.load(f)
        print("Loaded JSON data structure (first item):", data[0])
        print("Total number of items:", len(data))

    # Create the Sequential Memory Pile Dataset
    sequential_memory_dataset = SequentialMemoryPileDataset(
        data=data,
        num_samples=cfg["num_samples"]
    )

    # Create DataLoader
    sequential_data_loader = DataLoader(sequential_memory_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=default_collate)

    # Process data and save output
    processed_data = []

    for ar_data, seq2seq_data in sequential_data_loader:
        processed_data.append({
            'ar': ar_data['ar'],  # Directly use the raw text
            'seq2seq': seq2seq_data['seq2seq']  # Directly use the raw text from the previous AR
        })

    # Save the processed data to a JSON file
    with open(cfg["output_path"], "w") as output_file:
        json.dump(processed_data, output_file, indent=4)

    # Load the saved data
    with open(cfg["output_path"], "r") as output_file:
        loaded_data = json.load(output_file)

    # Randomly select two sequential examples
    random_index = random.randint(0, len(loaded_data) - 2)
    example_1 = loaded_data[random_index]
    example_2 = loaded_data[random_index + 1]

    # Debug: Print the structure of example_1 and example_2
    print("Example 1 structure:", example_1)
    print("Example 2 structure:", example_2)

    # Save the examples to a text file
    with open(cfg["detokenized_output_path"], "w") as detok_file:
        # Convert lists to strings by joining elements with a space (or appropriate delimiter)
        detok_file.write("Example 1 (AR text):\n")
        detok_file.write(" ".join(example_1['ar']) + "\n\n" if isinstance(example_1['ar'], list) else example_1['ar'] + "\n\n")
        
        detok_file.write("Example 1 (Seq2Seq text):\n")
        detok_file.write(" ".join(example_1['seq2seq']) + "\n\n" if isinstance(example_1['seq2seq'], list) else example_1['seq2seq'] + "\n\n")
        
        detok_file.write("Example 2 (AR text):\n")
        detok_file.write(" ".join(example_2['ar']) + "\n\n" if isinstance(example_2['ar'], list) else example_2['ar'] + "\n\n")
        
        detok_file.write("Example 2 (Seq2Seq text):\n")
        detok_file.write(" ".join(example_2['seq2seq']) + "\n" if isinstance(example_2['seq2seq'], list) else example_2['seq2seq'] + "\n")

if __name__ == "__main__":
    main()
