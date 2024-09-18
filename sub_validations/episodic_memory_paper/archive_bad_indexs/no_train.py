import os
import json
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AdamW

# Configuration dictionary
cfg = {
    "model_names": [
        "EleutherAI/pythia-14m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b"
    ],  # List of models
    "learning_rate": 1e-4,                  # Updated learning rate
    "training_batch_size": 1,
    "inference_batch_size": 200,  # Number of inferences done in parallel
    "max_inference_steps": 800,   # Total number of inference steps
    "num_train_epochs": 2,
    "pile_training_steps": 20,   # Number of steps to train on Pile data
    "pile_token_size": 2049,      # Size of tokens for Pile dataset
    "pile_inference_interval": 1, # Reinference every n steps on the Pile data
    "experiment_name": "no_train",
    "qa_data_file": "./qa_data.json",
    "pile_data_file": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_output.json",  # Full path to the Pile data
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.9,
        "top_k": 50,
        "do_sample": True,        # Use sampling-based generation mode
    },
    "gpu_device": "cuda:0",
    "skip_threshold": 0  # Skip threshold for correct counts
}

# Create experiment folder
def create_experiment_folder(cfg):
    experiment_folder = f"experiments/{cfg['experiment_name']}"
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

# Save the configuration to the experiment folder
def save_config(cfg, experiment_folder):
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(cfg, config_file, indent=4)

# Save results to CSV (Primary)
def save_results_to_csv(model_name, qa_index, num_epochs_trained, correct_count_single, correct_count_follow_up_single, experiment_folder):
    csv_path = os.path.join(experiment_folder, "results.csv")
    file_exists = os.path.isfile(csv_path)

    # Remove "EleutherAI/" from model name
    cleaned_model_name = model_name.replace("EleutherAI/", "")

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file does not exist
            writer.writerow(["Model Name", "QA Index", "Num Epochs Trained", "Correct Count QA", "Correct Count Related QA"])
        # Write results
        writer.writerow([cleaned_model_name, qa_index, num_epochs_trained, correct_count_single, correct_count_follow_up_single])

# Save results to CSV (Forgetting study)
def save_forgetting_results_to_csv(model_name, qa_index, step, correct_count_single, correct_count_follow_up_single, experiment_folder):
    csv_path = os.path.join(experiment_folder, "forgetting_results.csv")
    file_exists = os.path.isfile(csv_path)

    # Remove "EleutherAI/" from model name
    cleaned_model_name = model_name.replace("EleutherAI/", "")

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file does not exist
            writer.writerow(["Model Name", "QA Index", "Step", "Correct Count QA", "Correct Count Related QA"])
        # Write results
        writer.writerow([cleaned_model_name, qa_index, step, correct_count_single, correct_count_follow_up_single])

# Save step-wise loss during Pile training
def save_pile_training_loss(model_name, qa_index, step, loss, experiment_folder):
    csv_path = os.path.join(experiment_folder, "pile_training_loss.csv")
    file_exists = os.path.isfile(csv_path)

    # Remove "EleutherAI/" from model name
    cleaned_model_name = model_name.replace("EleutherAI/", "")

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file does not exist
            writer.writerow(["Model Name", "QA Index", "Step", "Pile Training Loss"])
        # Write step-wise loss results
        writer.writerow([cleaned_model_name, qa_index, step, loss])

# Custom Dataset class for QA data using a manual approach
class QADataset(Dataset):
    def __init__(self, qa_data, tokenizer, max_length=128):
        self.questions = qa_data['question']
        self.answers = qa_data['answer']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        text = f"Q: {question} A: {answer}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()  # Labels are the same as input_ids for causal LM
        }

# Dataset class for Pile data
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

class CustomTrainer:
    def __init__(self, cfg, qa_loader, tokenizer, model_name):
        self.cfg = cfg
        self.qa_loader = qa_loader
        self.device = cfg["gpu_device"]
        self.model_name = model_name  # Store model name for printout

        # Load the model and set pad_token_id to eos_token_id during model initialization
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(self.device)
        self.tokenizer = tokenizer

        # Add a `pad_token` if it does not exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        # Resize the model embeddings to account for the added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Set up the optimizer with fixed learning rate
        self.optimizer = AdamW(self.model.parameters(), lr=cfg["learning_rate"])

    def train(self, num_epochs):
        print(f"\nModel: {self.model_name}")  # Print model name before training starts

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for step, batch in enumerate(self.qa_loader):
                self.optimizer.zero_grad()

                # Move data to the GPU
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)

                # Forward pass, loss calculation, backward pass, and optimization
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update optimizer
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Print the training loss for each epoch
            avg_loss = total_loss / num_batches
            print(f"Epoch: {epoch + 1}, Training Loss: {avg_loss:.4f}")

    def inference(self, question, answer, params):
        self.model.eval()
        correct_count = 0

        # Total number of batches required for inference
        num_batches = self.cfg['max_inference_steps'] // self.cfg['inference_batch_size']

        with torch.no_grad():
            for _ in range(num_batches):
                batch_questions = [f"Q: {question} A:" for _ in range(self.cfg['inference_batch_size'])]

                # Prepare inputs and attention masks
                input_tokens = self.tokenizer(batch_questions, return_tensors='pt', padding=True).input_ids.to(self.device)
                attention_mask = (input_tokens != self.tokenizer.pad_token_id).to(self.device)

                # Generate response with the specified decoding parameters (using sampling mode)
                output = self.model.generate(
                    input_tokens,
                    attention_mask=attention_mask,
                    max_length=input_tokens.shape[1] + params['max_new_tokens'],
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    do_sample=params['do_sample'],  # Set sampling mode
                    pad_token_id=self.tokenizer.pad_token_id
                )

                generated_texts = [self.tokenizer.decode(output[i], skip_special_tokens=True) for i in range(self.cfg['inference_batch_size'])]

                # Check if the generated text contains the correct answer
                for generated_text in generated_texts:
                    if answer.lower() in generated_text.lower():
                        correct_count += 1

        return correct_count

    # Continued training on Pile data, reinferencing only on the single question trained initially
    def train_on_pile(self, pile_loader, qa_data, qa_index, num_steps, interval, experiment_folder, model_name):
        self.model.train()

        for step, batch in enumerate(pile_loader):
            if step >= num_steps:
                break

            self.optimizer.zero_grad()

            # Tokenize the raw input from Pile dataset before using
            input_tokens = self.tokenizer(batch['input_ids'], return_tensors='pt', padding=True).input_ids.to(self.device)
            labels = input_tokens

            # Forward pass, loss calculation, backward pass, and optimization
            outputs = self.model(input_ids=input_tokens, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update optimizer
            self.optimizer.step()

            # Save pile training loss to CSV
            save_pile_training_loss(model_name, qa_index + 1, step + 1, loss.item(), experiment_folder)

            # Reinference on the single QA and related QA after every `interval` steps
            if (step + 1) % interval == 0:
                # Only reinference on the specific QA index and its follow-up
                correct_count_single = self.inference(qa_data['qa_data']['question'][qa_index], qa_data['qa_data']['answer'][qa_index], self.cfg['inference_params'])
                correct_count_follow_up_single = self.inference(qa_data['follow_up_qa_data']['question'][qa_index], qa_data['follow_up_qa_data']['answer'][qa_index], self.cfg['inference_params'])

                # Print the reinference results
                print(f"Pile {step + 1}: Correct {correct_count_single} / {self.cfg['max_inference_steps']}, Related QA {qa_index + 1}: Correct {correct_count_follow_up_single} / {self.cfg['max_inference_steps']}")

                save_forgetting_results_to_csv(model_name, qa_index + 1, step + 1, correct_count_single, correct_count_follow_up_single, experiment_folder)

# Main function to run the experiment
def run_experiment(cfg):
    print(f"Running experiment with custom training and inference...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_names"][0])  # Use the first model's tokenizer

    # Load QA data from the JSON file
    with open(cfg['qa_data_file'], 'r') as f:
        qa_data = json.load(f)

    # Load Pile data from the JSON file
    with open(cfg['pile_data_file'], 'r') as f:
        detokenized_output = json.load(f)

    pile_dataset = PileDataset(detokenized_output)
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=True)

    # Create experiment folder and save config
    experiment_folder = create_experiment_folder(cfg)
    save_config(cfg, experiment_folder)

    # Iterate through each model
    for model_name in cfg['model_names']:
        print(f"\nRunning experiment for model: {model_name}")
        
        # Prepare dataset for manual training
        qa_dataset = QADataset(qa_data['qa_data'], tokenizer)
        qa_loader = DataLoader(qa_dataset, batch_size=cfg['training_batch_size'], shuffle=True)

        # Initialize custom trainer for the current model
        trainer = CustomTrainer(cfg, qa_loader, tokenizer, model_name)

        # Train on each single example with fresh model and inference
        for i in range(len(qa_data['qa_data']['question'])):
            # Reinitialize the model and optimizer for each individual example
            single_loader = DataLoader(QADataset({'question': [qa_data['qa_data']['question'][i]], 'answer': [qa_data['qa_data']['answer'][i]]}, tokenizer), batch_size=1)
            trainer = CustomTrainer(cfg, single_loader, tokenizer, model_name)
            
            # Train the model for the specified number of epochs
            #trainer.train(cfg['num_train_epochs'])

            # Inference on the trained question and its follow-up question
            correct_count_single = trainer.inference(qa_data['qa_data']['question'][i], qa_data['qa_data']['answer'][i], cfg['inference_params'])
            correct_count_follow_up_single = trainer.inference(qa_data['follow_up_qa_data']['question'][i], qa_data['follow_up_qa_data']['answer'][i], cfg['inference_params'])

            # Print combined results for QA and related QA
            print(f"QA {i + 1}: Correct {correct_count_single} / {cfg['max_inference_steps']}, Related QA {i + 1}: Correct {correct_count_follow_up_single} / {cfg['max_inference_steps']}")

            # Save results to CSV (with the QA index and number of epochs trained before inferencing)
            save_results_to_csv(model_name, i + 1, cfg['num_train_epochs'], correct_count_single, correct_count_follow_up_single, experiment_folder)

            # Continue training on Pile data and reinference the specific QA data
            trainer.train_on_pile(pile_loader, qa_data, i, cfg['pile_training_steps'], cfg['pile_inference_interval'], experiment_folder, model_name)

# Run the experiment
run_experiment(cfg)
