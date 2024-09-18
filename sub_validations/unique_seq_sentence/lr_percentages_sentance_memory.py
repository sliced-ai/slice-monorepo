import os
import json
import torch
import pandas as pd
import logging
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random

# Set environment variable to disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific warning
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define the configuration parameters
cfg = {
    "experiment_name": "2_example_context_smallerpile",  # Add your experiment name here
    "main_model": {
        "name": "EleutherAI/pythia-410m"
    },
    "batch_size": 1,
    "gpu_device": "cuda:0",
    "tokenizer_path": "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json",
    "num_epochs": 10,
    "step_size": 2,  # Step size for rolling window
    "window_size": 100,  # Number of examples in the rolling window
    "max_tokens": 3000,  # Maximum tokens in a batch
    "pile_token_size": 2049,  # Each pile data item has exactly 2049 tokens
    "learning_rates": [5e-4, 1e-4, 1e-5, 1e-6, 1e-7],  # Custom learning rates
    "percentages": [0.05, 0.05, 0.15, 0.25, 0.5],  # Updated percentages of parameters
    "word_level_copies": 2,  # Number of copies for word-level inference
    "token_level_copies": 2,  # Number of copies for token-level inference
    "output_csv_path": "inference_results.csv",  # Path to save the inference results
    "epoch_log_csv_path": "epoch_logs.csv",  # Path to save the epoch step logs
    "decoding_strategy": "beam_search",  # Decoding strategy: 'greedy', 'beam_search', 'top_k', 'top_p'
    "beam_width": 5,  # Beam width for beam search
    "top_k": 50,  # Top-k sampling
    "top_p": 0.9  # Top-p (nucleus) sampling
}

def create_experiment_folder(cfg):
    # Create a directory for the experiment
    experiment_dir = os.path.join(os.getcwd(), cfg["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)

    # Update paths in config to point to the experiment directory
    cfg["output_csv_path"] = os.path.join(experiment_dir, "inference_results.csv")
    cfg["epoch_log_csv_path"] = os.path.join(experiment_dir, "epoch_logs.csv")
    cfg["config_path"] = os.path.join(experiment_dir, "config.json")

    # Save a copy of the configuration to the experiment directory
    with open(cfg["config_path"], 'w') as f:
        json.dump(cfg, f, indent=4)

    return experiment_dir

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

        # Load the model and set pad_token_id to eos_token_id during model initialization
        self.model = GPTNeoXForCausalLM.from_pretrained(cfg["main_model"]["name"], pad_token_id=tokenizer.eos_token_id).to(self.device)
        self.tokenizer = tokenizer

        # Assign custom learning rates to parameter groups
        self.param_groups = self.assign_lr_to_params(self.model, cfg["learning_rates"], cfg["percentages"])
        self.optimizer = optim.AdamW(self.param_groups)

        self.num_epochs = cfg["num_epochs"]

        # Initialize DataFrames to save results and logs
        self.results_df = pd.DataFrame(columns=["epoch", "step", "inference_type", "context", "prediction", "target", "correct"])
        self.epoch_log_df = pd.DataFrame(columns=["epoch", "step", "train_loss", "token_correct", "word_correct", "total_predictions"])

    def assign_lr_to_params(self, model, learning_rates, percentages):
        # Get all parameters and shuffle them randomly
        all_params = list(model.named_parameters())
        random.shuffle(all_params)

        param_groups = []
        start_idx = 0

        for lr, percentage in zip(learning_rates, percentages):
            num_params = int(len(all_params) * percentage)
            param_group = [param for name, param in all_params[start_idx:start_idx + num_params]]
            param_groups.append({"params": param_group, "lr": lr})
            start_idx += num_params

        return param_groups

    def train(self):
        self.model.train()

        step_number = 0  # Initialize step number

        for epoch in range(self.num_epochs):
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

                # Create attention mask
                attention_mask = torch.ones_like(combined_input_ids, device=self.device)

                # Create batch input
                batch_inputs = {'input_ids': combined_input_ids, 'attention_mask': attention_mask}

                # Forward pass, backward pass, and optimization
                self.optimizer.zero_grad()
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # Perform evaluation after training step and save results to CSV
                token_correct, word_correct, total = self.evaluate_and_save_results(rw_batch, epoch + 1, step_number)

                # Print epoch, step, loss, and evaluation results in one line
                print(f"Epoch {epoch + 1}, Step {step_number}, Train Loss: {loss.item()}, Token Prediction: {token_correct}/{total}, Word Prediction: {word_correct}/{total}")

                # Save epoch step logs to DataFrame
                log_row = pd.DataFrame([{
                    "epoch": epoch + 1,
                    "step": step_number,
                    "train_loss": loss.item(),
                    "token_correct": token_correct,
                    "word_correct": word_correct,
                    "total_predictions": total
                }])
                self.epoch_log_df = pd.concat([self.epoch_log_df, log_row], ignore_index=True)

        # Save the epoch logs to a CSV file
        self.epoch_log_df.to_csv(self.cfg["epoch_log_csv_path"], index=False)

    def evaluate_and_save_results(self, rw_batch, epoch, step):
        # Word-Level Evaluation and Saving
        word_correct, total_word = self.evaluate_and_save(rw_batch, epoch, step, "word")

        # Token-Level Evaluation and Saving
        token_correct, total_token = self.evaluate_and_save(rw_batch, epoch, step, "token")

        # Save the DataFrame to CSV
        self.results_df.to_csv(self.cfg["output_csv_path"], index=False)

        return token_correct, word_correct, total_word

    def evaluate_and_save(self, rw_batch, epoch, step, inference_type):
        correct_predictions = 0
        sentences = rw_batch['input_ids'][0].split("\n")
        total_predictions = len(sentences)

        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue  # Skip if the sentence is too short

            if inference_type == "word":
                input_sentence = " ".join(words[:-1])
                target = words[-1]
                target_tokens = self.tokenizer(target, return_tensors='pt').input_ids.squeeze()
                context_sentences = ("\n".join([sentence] * self.cfg["word_level_copies"]) + "\n" + input_sentence)
            else:
                tokens = self.tokenizer(sentence, return_tensors='pt').input_ids
                target = self.tokenizer.decode(tokens[0, -1])
                target_tokens = tokens[0, -1]
                context_sentences = ("\n".join([sentence] * self.cfg["token_level_copies"]) + "\n" + self.tokenizer.decode(tokens[0, :-1]))

            # Tokenize and predict
            input_tokens = self.tokenizer(context_sentences, return_tensors='pt', padding=False).input_ids.to(self.device)

            # Create attention mask for inference
            attention_mask = torch.ones_like(input_tokens, device=self.device)

            # Fix IndexError by ensuring target_tokens is treated as a tensor with correct dimensions
            max_length = input_tokens.size(1) + (target_tokens.size(0) if target_tokens.dim() > 0 else 1)

            predicted_token = None
            predicted_word = None

            with torch.no_grad():
                if self.cfg["decoding_strategy"] == "greedy":
                    # Greedy decoding (simplest)
                    output = self.model(input_tokens, attention_mask=attention_mask).logits
                    missing_position = input_tokens.size(1) - 1
                    predicted_token = torch.argmax(output[-1, missing_position, :], dim=-1)
                    predicted_word = self.tokenizer.decode(predicted_token)

                elif self.cfg["decoding_strategy"] == "beam_search":
                    # Beam search decoding
                    output = self.model.generate(input_tokens, attention_mask=attention_mask, max_length=max_length, num_beams=self.cfg["beam_width"], early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                    predicted_word = self.tokenizer.decode(output[:, input_tokens.size(1):].squeeze())

                elif self.cfg["decoding_strategy"] == "top_k":
                    # Top-k sampling
                    output = self.model.generate(input_tokens, attention_mask=attention_mask, max_length=max_length, do_sample=True, top_k=self.cfg["top_k"], pad_token_id=self.tokenizer.pad_token_id)
                    predicted_word = self.tokenizer.decode(output[:, input_tokens.size(1):].squeeze())

                elif self.cfg["decoding_strategy"] == "top_p":
                    # Top-p (nucleus) sampling
                    output = self.model.generate(input_tokens, attention_mask=attention_mask, max_length=max_length, do_sample=True, top_p=self.cfg["top_p"], pad_token_id=self.tokenizer.pad_token_id)
                    predicted_word = self.tokenizer.decode(output[:, input_tokens.size(1):].squeeze())

            # Check if prediction is correct
            if inference_type == "word":
                # Allow predictions that start with the correct target (ignoring extra tokens)
                correct = predicted_word.strip().startswith(target)
            else:
                # Handle token-level prediction, ignoring leading/trailing whitespace
                correct = predicted_word.strip() == target.strip()

            if correct:
                correct_predictions += 1

            # Append the result to the DataFrame using pandas.concat()
            result_row = pd.DataFrame([{
                "epoch": epoch,
                "step": step,
                "inference_type": inference_type,
                "context": context_sentences,
                "prediction": predicted_word.strip(),
                "target": target,
                "correct": correct
            }])
            self.results_df = pd.concat([self.results_df, result_row], ignore_index=True)

            # Free up GPU memory by deleting only allocated variables
            del input_tokens, attention_mask, output
            torch.cuda.empty_cache()

        return correct_predictions, total_predictions

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"], clean_up_tokenization_spaces=False)

    # Load datasets from JSON files using the standard JSON library
    synthetic_sentences = load_json_file('synthetic_sentences_100k.json')
    detokenized_output = load_json_file('detokenized_pile_1M.json')

    # Create the experiment folder and save config
    create_experiment_folder(cfg)

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
