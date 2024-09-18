import os
import json
import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import Dataset
import torch.optim as optim
import csv
from collections import deque

# Configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-70m"
    },
    "experiment_name": "pile_training",
    "starting_learning_rate": 1e-5,
    "batch_size": 1,  # We'll handle multiple pile items manually
    "gpu_device": "cuda:2",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "num_epochs": 5,  # Set to 5 epochs
    "max_tokens": 4098,  # Max tokens per batch
    "pile_token_size": 2049,  # Each pile data item has 2049 tokens
    "json_file_path": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_pile_1M.json"
}

class PileDataset(Dataset):
    """Dataset for loading detokenized Pile data."""
    def __init__(self, detokenized_texts):
        self.detokenized_texts = detokenized_texts

    def __len__(self):
        return len(self.detokenized_texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.detokenized_texts[idx],
            'index': idx
        }

class PileTrainer:
    """Trainer for fine-tuning LLM on Pile data with dynamic token batching."""
    def __init__(self, cfg, pile_dataset, tokenizer):
        self.cfg = cfg
        self.pile_dataset = pile_dataset
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        self.tokenizer = tokenizer
        self.loss_history = []  # To store losses for CSV
        self.rolling_losses_50 = deque(maxlen=50)  # Rolling average over 50 steps
        self.rolling_losses_250 = deque(maxlen=250)  # Rolling average over 250 steps

        # Create directory to save the experiment files based on model name (single level)
        self.exp_dir = cfg["main_model"]["name"]
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save configuration to experiment folder
        self.cfg_file_path = os.path.join(self.exp_dir, "config.json")
        with open(self.cfg_file_path, 'w') as cfg_file:
            json.dump(cfg, cfg_file, indent=4)

        # Prepare CSV file path for saving training logs
        self.csv_file_path = os.path.join(self.exp_dir, "training_loss_log.csv")

    def train(self):
        """Main training loop with dynamic token batching and epoch control."""
        self.model.train()

        total_pile_items = len(self.pile_dataset)
        current_index = 0

        # Open CSV file to log losses
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Step", "Training Loss"])  # Write CSV headers

            for epoch in range(self.cfg["num_epochs"]):
                print(f"Starting Epoch {epoch + 1}/{self.cfg['num_epochs']}")
                step = 0

                while current_index < total_pile_items:
                    # Collect tokens until we hit the max token limit
                    pile_tokens_list = []
                    token_count = 0

                    # Accumulate tokens until the max token limit is reached
                    while token_count < self.cfg["max_tokens"] and current_index < total_pile_items:
                        pile_item = self.pile_dataset[current_index]
                        pile_item_tokens = self.tokenizer(pile_item['input_ids'], return_tensors='pt', padding=False, truncation=False).input_ids.to(self.device)
                        pile_tokens_list.append(pile_item_tokens)
                        token_count += pile_item_tokens.size(1)
                        current_index += 1

                    # Concatenate the tokenized pile items
                    combined_pile_tokens = torch.cat(pile_tokens_list, dim=1)

                    # If combined tokens exceed the max limit, truncate them
                    if combined_pile_tokens.size(1) > self.cfg["max_tokens"]:
                        combined_pile_tokens = combined_pile_tokens[:, :self.cfg["max_tokens"]]

                    # Prepare batch input
                    batch_inputs = {'input_ids': combined_pile_tokens}

                    # Forward pass, backward pass, and optimization
                    self.optimizer.zero_grad()
                    outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()

                    # Log training loss
                    step += 1
                    self.loss_history.append((step, loss.item()))
                    self.rolling_losses_50.append(loss.item())
                    self.rolling_losses_250.append(loss.item())

                    # Write the current step and loss to the CSV file
                    writer.writerow([epoch + 1, step, loss.item()])

                    # Calculate rolling average loss over 50 and 250 steps
                    rolling_avg_50 = sum(self.rolling_losses_50) / len(self.rolling_losses_50)
                    rolling_avg_250 = sum(self.rolling_losses_250) / len(self.rolling_losses_250)

                    # Print the training loss and rolling averages for each step
                    print(f"Epoch {epoch + 1}, Step {step}, Train Loss: {loss.item()}, "
                          f"Rolling Avg (Last 50): {rolling_avg_50:.4f}, Rolling Avg (Last 250): {rolling_avg_250:.4f}")

                    # Flush the CSV file so it gets updated after every batch
                    file.flush()

                print(f"Finished Epoch {epoch + 1}, Processed {current_index}/{total_pile_items} Pile Items")

                # Save the model after every epoch
                model_save_path = os.path.join(self.exp_dir, f"model_epoch_{epoch + 1}.bin")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Model saved after Epoch {epoch + 1} at {model_save_path}")

def load_json_file(file_path):
    """Load a JSON file from a given path."""
    with open(file_path, 'r') as f:
        data = json.load(f)  # Load JSON data
    return data

def main():
    """Main function to initialize and run the training process."""
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)
    
    # Load the Pile dataset from the specified JSON file
    detokenized_output = load_json_file(cfg["json_file_path"])

    # Create the Pile dataset
    pile_dataset = PileDataset(detokenized_output)

    # Initialize and start the training process
    trainer = PileTrainer(cfg, pile_dataset, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
