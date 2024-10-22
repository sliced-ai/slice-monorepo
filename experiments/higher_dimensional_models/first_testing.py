import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import GPTNeoXForCausalLM, GPT2TokenizerFast, get_linear_schedule_with_warmup
from torch.nn import functional as F
import math

# Configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-14m"
    },
    "experiment_name": "custom_neuron_experiment_v4",
    "starting_learning_rate": 1e-5,
    "batch_size": 1,  # Batch size set to 1
    "gpu_device": "cuda" if torch.cuda.is_available() else "cpu",
    "tokenizer_name": "gpt2",  # Using an off-the-shelf tokenizer
    "experiments_dir": "./experiments",
    "num_epochs": 1,
    "sequential_data_path": "detokenized_output.json",  # Updated data path
    "max_steps_per_epoch": None,
    "max_tokens": 1024  # Maximum tokens in a batch
}

# Ensure experiments directory exists
os.makedirs(cfg["experiments_dir"], exist_ok=True)

# Define the custom neuron module by subclassing nn.Linear
class CustomNeuronLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_comp_neurons=2, bias=True):
        super(CustomNeuronLinear, self).__init__(in_features, out_features, bias)
        self.num_comp_neurons = num_comp_neurons

        # Selection neuron outputs selection logits for each output neuron
        self.selection_layer = nn.Linear(in_features, out_features * num_comp_neurons)

        # Computation neurons: Initialize weights for each computation neuron
        # Shape: [num_comp_neurons, out_features, in_features]
        self.comp_weights = nn.Parameter(torch.Tensor(num_comp_neurons, out_features, in_features))
        if bias:
            self.comp_biases = nn.Parameter(torch.Tensor(num_comp_neurons, out_features))
        else:
            self.comp_biases = None

        # Initialize computation neurons with the original weights
        self.initialize_computation_neurons()

    def initialize_computation_neurons(self):
        # Initialize computation neurons with the pretrained weights
        for i in range(self.num_comp_neurons):
            self.comp_weights.data[i] = self.weight.data.clone()
            if self.bias is not None:
                self.comp_biases.data[i] = self.bias.data.clone()

    def forward(self, input):
        # input shape: [batch_size, seq_len, in_features]
        batch_size, seq_len, in_features = input.size()
    
        # Compute selection logits
        selection_logits = self.selection_layer(input)
        selection_logits = selection_logits.view(batch_size, seq_len, self.out_features, self.num_comp_neurons)
    
        # Compute selection probabilities
        selection_probs = F.softmax(selection_logits, dim=-1)
    
        # Hard selection using argmax
        with torch.no_grad():
            selected_idx = torch.argmax(selection_probs, dim=-1)
            selected_mask_hard = F.one_hot(selected_idx, num_classes=self.num_comp_neurons).float()
    
        # Straight-through estimator
        selected_mask = selected_mask_hard - selection_probs.detach() + selection_probs
    
        # Compute outputs from all computation neurons
        # input: [batch_size, seq_len, in_features]
        # comp_weights: [num_comp_neurons, out_features, in_features]
    
        # Reshape comp_weights to [num_comp_neurons, in_features, out_features]
        comp_weights = self.comp_weights.permute(0, 2, 1)  # Now shape: [num_comp_neurons, in_features, out_features]
    
        # Expand input for batched computation
        # Input shape: [batch_size, seq_len, in_features]
        # We need to compute the outputs for each computation neuron
        # So we expand input to [batch_size, seq_len, 1, in_features]
        input_expanded = input.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, in_features]
    
        # Compute outputs: batch matrix multiplication
        # comp_weights: [num_comp_neurons, in_features, out_features]
        # We need to expand comp_weights to [1, 1, num_comp_neurons, in_features, out_features]
        comp_weights_expanded = comp_weights.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_comp_neurons, in_features, out_features]
    
        # Perform batch matrix multiplication
        # input_expanded: [batch_size, seq_len, 1, in_features]
        # comp_weights_expanded: [1, 1, num_comp_neurons, in_features, out_features]
        # Resulting comp_outputs: [batch_size, seq_len, num_comp_neurons, out_features]
        comp_outputs = torch.matmul(input_expanded, comp_weights_expanded).squeeze(3)  # Remove the singleton dimension
    
        # Add biases if present
        if self.comp_biases is not None:
            comp_biases_expanded = self.comp_biases.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_comp_neurons, out_features]
            comp_outputs = comp_outputs + comp_biases_expanded
    
        # Apply selected_mask
        # selected_mask: [batch_size, seq_len, out_features, num_comp_neurons]
        # Transpose selected_mask to align dimensions
        selected_mask = selected_mask.permute(0, 1, 3, 2)  # Shape: [batch_size, seq_len, num_comp_neurons, out_features]
    
        # Element-wise multiplication and sum over num_comp_neurons dimension
        output = torch.sum(comp_outputs * selected_mask, dim=2)  # Shape: [batch_size, seq_len, out_features]
    
        return output


# Function to replace applicable Linear layers with CustomNeuronLinear
def replace_linear_with_custom_neuron(module, num_comp_neurons=2):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Only replace Linear layers that are part of the transformer blocks
            # Avoid replacing embedding layers or LayerNorm layers
            custom_layer = CustomNeuronLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                num_comp_neurons=num_comp_neurons,
                bias=child.bias is not None
            )

            # Initialize computation neurons with the weights of the original Linear layer
            custom_layer.weight = child.weight
            if child.bias is not None:
                custom_layer.bias = child.bias

            # Replace the Linear layer with CustomNeuronLinear
            setattr(module, name, custom_layer)
            print(f"Modified nn.Linear layer '{name}' to include custom neurons.")
        else:
            # Recursively apply to child modules
            replace_linear_with_custom_neuron(child, num_comp_neurons)

# Dataset class for sequential data
class SimpleJSONDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=1024):
        """
        Args:
            json_path (str): Path to the JSON file containing text data.
            tokenizer (PreTrainedTokenizer): Tokenizer to encode the text.
            max_length (int): Maximum token length.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # No padding since batch size is 1
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Shape: [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: [seq_len]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# Trainer class
class CustomNeuronTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["gpu_device"]

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(cfg["tokenizer_name"])
        # Fix tokenizer padding issue
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pretrained model
        print("Loading pretrained model...")
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"])

        # Modify Linear layers to include custom neurons
        print("Modifying Linear layers to include custom neurons...")
        replace_linear_with_custom_neuron(self.model, num_comp_neurons=2)

        # Move model to device
        self.model.to(self.device)

        # Create dataset
        self.prepare_data()

        # Setup optimizer and scheduler
        total_steps = len(self.dataset) * cfg["num_epochs"]
        warmup_steps = int(total_steps * 0.1) if total_steps > 0 else 0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    def prepare_data(self):
        cfg = self.cfg
        # Sequential data
        print("Preparing dataset...")
        self.dataset = SimpleJSONDataset(
            json_path=cfg["sequential_data_path"],
            tokenizer=self.tokenizer,
            max_length=cfg["max_tokens"]
        )
        print(f"Dataset prepared with {len(self.dataset)} samples.")

    def train(self):
        self.model.train()
        cfg = self.cfg
        total_steps = len(self.dataset) * cfg["num_epochs"]
        step = 0

        for epoch in range(cfg["num_epochs"]):
            print(f"\nStarting Epoch {epoch + 1}/{cfg['num_epochs']}")
            epoch_loss = 0.0
            for idx in range(len(self.dataset)):
                step += 1
                batch = self.dataset[idx]
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)  # Shape: [1, seq_len]
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)  # Shape: [1, seq_len]
                labels = input_ids.clone()

                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                except RuntimeError as e:
                    print(f"RuntimeError during model forward pass: {e}")
                    raise

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                if step % 10 == 0:
                    print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(self.dataset)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

            # Save model checkpoint
            checkpoint_path = os.path.join(cfg["experiments_dir"], f"{cfg['experiment_name']}_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        print("\nTraining completed.")

# Main function
def main():
    trainer = CustomNeuronTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
