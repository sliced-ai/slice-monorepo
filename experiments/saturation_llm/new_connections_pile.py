import os
import json
import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import Dataset
import torch.optim as optim
import random

# Configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-70m-deduped"
    },
    "starting_learning_rate": 1e-5,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "num_epochs": 5,
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

# Hook outputs
hook_outputs = {}

def hook_fn(module, input, output):
    """Hook function to capture outputs from a module"""
    layer_name = module.__class__.__name__
    if isinstance(output, tuple):
        output = output[0]  # Use the first element if it's a tuple (common in transformer outputs)
    hook_outputs[layer_name] = output

class CustomGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """Customized GPT-NeoX model with dynamic jump connections using hooks."""
    def __init__(self, config):
        super().__init__(config)
        self.active_connections = []
        self.total_jump_connections = 0  # To track the total number of connections

    def apply_hooks(self):
        """Apply hooks to the model layers to capture activations"""
        for idx, block in enumerate(self.gpt_neox.layers):
            block.attention.register_forward_hook(hook_fn)
            block.mlp.register_forward_hook(hook_fn)

    def add_jump_connection(self):
        """Adds a dynamic vertical jump connection using hook outputs"""
        if "GPTNeoXAttention" in hook_outputs and "GPTNeoXMLP" in hook_outputs:
            attn_output = hook_outputs["GPTNeoXAttention"]
            mlp_output = hook_outputs["GPTNeoXMLP"]
            
            # Randomly select neurons to connect
            neuron_1 = torch.randint(0, attn_output.size(-1), (1,)).item()
            neuron_2 = torch.randint(0, mlp_output.size(-1), (1,)).item()

            print(f"Jumping from Attention Neuron {neuron_1} to MLP Neuron {neuron_2}")

            # Perform the jump connection (adding attention neuron activation to MLP neuron)
            mlp_output[:, :, neuron_2] += attn_output[:, :, neuron_1]

            # Log the jump connection
            self.active_connections.append((neuron_1, neuron_2))
            self.total_jump_connections += 1

            print(f"Jump connection created between Attention Neuron {neuron_1} and MLP Neuron {neuron_2}")
            return True
        else:
            print("Required layer outputs not found for jump connection.")
            return False

class PileTrainer:
    """Trainer for fine-tuning LLM on Pile data with dynamic token batching."""
    def __init__(self, cfg, pile_dataset, tokenizer):
        self.cfg = cfg
        self.pile_dataset = pile_dataset
        self.device = cfg["gpu_device"]
        self.model = CustomGPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.model.apply_hooks()  # Apply hooks to capture activations
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        self.tokenizer = tokenizer

    def train(self):
        """Main training loop with one example per batch."""
        self.model.train()

        total_pile_items = len(self.pile_dataset)
        current_index = 0

        for epoch in range(self.cfg["num_epochs"]):
            print(f"Starting Epoch {epoch + 1}/{self.cfg['num_epochs']}")
            step = 0

            while current_index < total_pile_items:
                pile_item = self.pile_dataset[current_index]
                pile_item_tokens = self.tokenizer(pile_item['input_ids'], return_tensors='pt', padding=False, truncation=False).input_ids.to(self.device)
                current_index += 1

                # Prepare batch input
                batch_inputs = {'input_ids': pile_item_tokens}

                # Forward pass, backward pass, and optimization
                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # After the forward pass, apply a jump connection
                self.model.add_jump_connection()

                # Print loss and total connections
                step += 1
                print(f"Epoch {epoch + 1}, Step {step}, Train Loss: {loss.item()}, Total Jump Connections: {self.model.total_jump_connections}")

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
