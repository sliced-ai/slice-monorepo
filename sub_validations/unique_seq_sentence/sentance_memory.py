import os
import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import orjson  # Use orjson for faster JSON loading

# Set environment variable to disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the configuration parameters
cfg = {
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "experiment_name": "combined_training_7",
    "starting_learning_rate": 5e-5,  # Updated learning rate
    "batch_size": 1,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "num_epochs": 10,
    "step_size": 2,  # Step size for rolling window
    "window_size": 100,  # Number of examples in the rolling window
    "max_tokens": 3800,  # Maximum tokens in a batch
    "pile_token_size": 2049  # Each pile data item has exactly 2049 tokens
}

class RollingWindowDataset(Dataset):
    def __init__(self, sentences, window_size, step_size):
        self.sentences = sentences
        self.window_size = window_size
        self.step_size = step_size
        self.num_samples = len(self.sentences) - window_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.window_size
        sentence_window = self.sentences[start_idx:end_idx]
        sentence_str = "\n".join([s['sentence'] for s in sentence_window])  # Add newline between sentences
        return {
            'input_ids': sentence_str,
            'index': idx
        }

class PileDataset(Dataset):
    def __init__(self, detokenized_texts):
        self.detokenized_texts = detokenized_texts

    def __len__(self):
        return len(self.detokenized_texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.detokenized_texts[idx],
            'index': idx
        }

class CombinedTrainer:
    def __init__(self, cfg, rw_loader, pile_loader, tokenizer):
        self.cfg = cfg
        self.rw_loader = rw_loader
        self.pile_loader = pile_loader
        self.device = cfg["gpu_device"]
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["starting_learning_rate"])
        self.tokenizer = tokenizer

    def train(self):
        self.model.train()

        step_number = 0  # Initialize step number

        for epoch in range(self.cfg["num_epochs"]):
            print(f"Starting Epoch {epoch + 1}/{self.cfg['num_epochs']}")
            rw_iter = iter(self.rw_loader)
            pile_iter = iter(self.pile_loader)

            while True:
                step_number += 1  # Increment step number
                # Get the next rolling window data
                rw_batch = next(rw_iter, None)
                if rw_batch is None:
                    break  # End of the rolling window dataset

                # Tokenize the rolling window batch without truncation
                rw_tokens = self.tokenizer(rw_batch['input_ids'], return_tensors='pt', padding=False, truncation=False).input_ids
                rw_token_count = rw_tokens.size(1)

                # Start with one full pile data item
                pile_batch = next(pile_iter)
                pile_tokens = self.tokenizer(pile_batch['input_ids'], return_tensors='pt', padding=False, truncation=False).input_ids

                # Calculate the remaining token space after adding the rolling window data
                remaining_tokens = self.cfg["max_tokens"] - rw_token_count
                pile_token_count = min(remaining_tokens, self.cfg["pile_token_size"])

                # Check if we need additional pile data
                if pile_token_count < remaining_tokens:
                    # Add more pile data to fill up the remaining space
                    next_pile_batch = next(pile_iter)
                    next_pile_tokens = self.tokenizer(next_pile_batch['input_ids'], return_tensors='pt', padding=False, truncation=False).input_ids
                    next_pile_tokens = next_pile_tokens[:, :remaining_tokens - pile_token_count]  # Clip to fit remaining tokens
                    pile_tokens = torch.cat([pile_tokens, next_pile_tokens], dim=1)

                # Add dividing text between rolling window and pile data
                dividing_token = self.tokenizer("#####", return_tensors='pt').input_ids
                combined_input_ids = torch.cat([rw_tokens, dividing_token, pile_tokens], dim=1).to(self.device)

                # Create batch input
                batch_inputs = {'input_ids': combined_input_ids}

                # Forward pass, backward pass, and optimization
                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # Perform evaluation after training step
                token_correct, word_correct, total = self.evaluate_rolling_window(rw_batch)

                # Print epoch, step, loss, and evaluation results in one line
                print(f"Epoch {epoch + 1}, Step {step_number}, Train Loss: {loss.item()}, Token Prediction: {token_correct}/{total}, Word Prediction: {word_correct}/{total}")

    def evaluate_rolling_window(self, rw_batch):
        correct_token_predictions = 0
        correct_word_predictions = 0

        # Split the single string in rw_batch['input_ids'] by newline to get individual sentences
        sentences = rw_batch['input_ids'][0].split("\n")
        total_predictions = len(sentences)

        # Word-Level Evaluation
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue  # Skip if the sentence is too short

            input_sentence = " ".join(words[:-1])
            target_word = words[-1]

            input_tokens = self.tokenizer(input_sentence, return_tensors='pt').input_ids.to(self.device)

            with torch.no_grad():
                output = self.model(input_tokens).logits
                predicted_token = torch.argmax(output[:, -1, :], dim=-1)

            predicted_word = self.tokenizer.decode(predicted_token)

            if predicted_word.strip() == target_word:
                correct_word_predictions += 1

        # Token-Level Evaluation
        for sentence in sentences:
            tokens = self.tokenizer(sentence, return_tensors='pt').input_ids.to(self.device)
            target_token = tokens[:, -1]  # Last word token

            input_tokens = tokens[:, :-1]  # Remove the last word token

            with torch.no_grad():
                output = self.model(input_tokens).logits
                predicted_token = torch.argmax(output[:, -1, :], dim=-1)

            if predicted_token == target_token:
                correct_token_predictions += 1

        return correct_token_predictions, correct_word_predictions, total_predictions

def load_json_file(file_path):
    with open(file_path, 'rb') as f:  # Open in binary mode
        data = orjson.loads(f.read())  # Load using orjson for speed
    return data

def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)
    
    # Load datasets from JSON files using orjson for faster loading
    synthetic_sentences = load_json_file('synthetic_sentences_100k.json')
    detokenized_output = load_json_file('detokenized_pile_1M.json')

    # Create datasets and data loaders
    rw_dataset = RollingWindowDataset(synthetic_sentences, window_size=cfg["window_size"], step_size=cfg["step_size"])
    rw_loader = DataLoader(rw_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)  # Shuffle set to False

    pile_dataset = PileDataset(detokenized_output)
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)  # Shuffle set to False

    # Initialize and start training
    trainer = CombinedTrainer(cfg, rw_loader, pile_loader, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
