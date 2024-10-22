import os
import json
import pandas as pd
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
    ],
    "learning_rate": 1e-4,
    "training_batch_size": 1,
    "inference_batch_size": 200,
    "max_inference_steps": 400,
    "num_train_epochs": 2,
    "pile_training_steps": 50,
    "pile_inference_interval": 1,
    "experiment_name": "optimizer_fix",
    "qa_data_file": "/workspace/slice-monorepo/sub_validations/episodic_memory_paper/qa_data.json",
    "pile_data_file": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_output.json",
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True,
    },
    "gpu_device": "cuda:0"
}

# Create experiment folder
def create_experiment_folder(cfg):
    experiment_folder = f"experiments/{cfg['experiment_name']}"
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

# Batch saving of CSV data using Pandas
class CSVLogger:
    def __init__(self, file_path, columns):
        self.file_path = file_path
        self.columns = columns
        self.buffer = []

    # Accumulate rows in the buffer
    def log(self, row):
        self.buffer.append(row)

    # Write the buffer to CSV in bulk
    def flush(self):
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=self.columns)
            if not os.path.exists(self.file_path):
                df.to_csv(self.file_path, mode='w', header=True, index=False)
            else:
                df.to_csv(self.file_path, mode='a', header=False, index=False)
            self.buffer = []  # Clear the buffer after writing

# Save the configuration to the experiment folder
def save_config(cfg, experiment_folder):
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(cfg, config_file, indent=4)

# Custom Dataset class for QA data
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
            'labels': tokenized['input_ids'].squeeze()
        }

# Dataset class for Pile data
class PileDataset(Dataset):
    def __init__(self, detokenized_texts, tokenizer, max_length=128):
        self.detokenized_texts = detokenized_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.detokenized_texts)

    def __getitem__(self, idx):
        text = self.detokenized_texts[idx]
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze()
        }

class CustomTrainer:
    def __init__(self, cfg, qa_loader, tokenizer, model_name, experiment_folder):
        self.cfg = cfg
        self.qa_loader = qa_loader
        self.device = cfg["gpu_device"]
        self.model_name = model_name
        self.experiment_folder = experiment_folder
        self.training_loss_logger = CSVLogger(os.path.join(experiment_folder, "training_loss.csv"), 
                                              ["Model Name", "Trained Question", "Step", "Loss", "Type", "Inference Loss"])
        self.inference_results_logger = CSVLogger(os.path.join(experiment_folder, "inference_results.csv"), 
                                                  ["Model Name", "Trained", "Type", "Step", "Question", "Answer", "Correct", "Raw Output"])

        # Load the model and set pad_token_id to eos_token_id during model initialization
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(self.device)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self, num_epochs, trained_question):
        print(f"\nModel: {self.model_name}")
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=1e-4)
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for step, batch in enumerate(self.qa_loader):
                optimizer.zero_grad()

                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            # Perform a forward pass for the trained question (inference loss) without training
            inference_loss = self.get_inference_loss(trained_question, labels)

            print(f"Epoch: {epoch + 1}, Training Loss: {avg_loss:.4f}, Inference Loss: {inference_loss:.4f}")
            self.training_loss_logger.log([self.model_name, trained_question, epoch + 1, avg_loss, "no_pile", inference_loss])
            self.training_loss_logger.flush()  # Ensure the buffer is flushed after every epoch

    def get_inference_loss(self, question, labels):
        """ Get loss from a forward pass (without updating model weights). """
        self.model.eval()  # Put model in eval mode
        with torch.no_grad():
            tokenized = self.tokenizer(f"Q: {question} A:", truncation=True, padding='max_length', max_length=128, return_tensors="pt")
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
        return loss

    def inference(self, question, answer, params, trained_question, step, inference_type):
        self.model.eval()
        correct_count = 0
        raw_outputs = []

        num_batches = self.cfg['max_inference_steps'] // self.cfg['inference_batch_size']

        with torch.no_grad():
            for _ in range(num_batches):
                batch_questions = [f"Q: {question} A:" for _ in range(self.cfg['inference_batch_size'])]

                input_tokens = self.tokenizer(batch_questions, return_tensors='pt', padding=True).input_ids.to(self.device)
                attention_mask = (input_tokens != self.tokenizer.pad_token_id).to(self.device)

                output = self.model.generate(
                    input_tokens,
                    attention_mask=attention_mask,
                    max_length=input_tokens.shape[1] + params['max_new_tokens'],
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    do_sample=params['do_sample'],
                    pad_token_id=self.tokenizer.pad_token_id
                )

                generated_texts = [self.tokenizer.decode(output[i], skip_special_tokens=True) for i in range(self.cfg['inference_batch_size'])]

                for generated_text in generated_texts:
                    raw_outputs.append(generated_text)
                    correct = answer.lower() in generated_text.lower()  # Check if the correct answer is present
                    self.inference_results_logger.log([self.model_name, trained_question, inference_type, step, question, answer, correct, generated_text])
                    if correct:
                        correct_count += 1

        self.inference_results_logger.flush()  # Ensure logs are saved after each inference
        return correct_count, raw_outputs

    def train_on_pile(self, pile_loader, qa_data, qa_index, num_steps, interval, trained_question):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=1e-4)
        # Re-tokenize the trained question to use the correct labels during pile training inference loss
        tokenized_trained_question = self.tokenizer(f"Q: {trained_question} A:", truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        labels_for_trained_question = tokenized_trained_question['input_ids'].to(self.device)

        for step, batch in enumerate(pile_loader):
            if step >= num_steps:
                break

            optimizer.zero_grad()

            # Convert pile dataset batch into tensors
            input_tokens = batch['input_ids'].to(self.device)
            labels = batch['input_ids'].to(self.device)

            outputs = self.model(input_ids=input_tokens, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Perform a forward pass for the trained question (inference loss) after pile training step
            inference_loss = self.get_inference_loss(trained_question, labels_for_trained_question)

            print(f"Pile {step + 1}: Training Loss: {loss.item():.4f}, Inference Loss: {inference_loss:.4f}")
            self.training_loss_logger.log([self.model_name, trained_question, step + 1, loss.item(), "pile", inference_loss])

            if (step + 1) % interval == 0:
                # Inference on QA and related QA during pile training
                correct_count_single = self.inference(
                    qa_data['qa_data']['question'][qa_index],
                    qa_data['qa_data']['answer'][qa_index],
                    self.cfg['inference_params'],
                    qa_data['qa_data']['question'][qa_index],
                    step + 1,
                    "pile"
                )[0]
                correct_count_follow_up = self.inference(
                    qa_data['follow_up_qa_data']['question'][qa_index],
                    qa_data['follow_up_qa_data']['answer'][qa_index],
                    self.cfg['inference_params'],
                    qa_data['qa_data']['question'][qa_index],
                    step + 1,
                    "pile"
                )[0]
                print(f"Pile {step + 1}: Correct {correct_count_single} / 10, Related QA {qa_index + 1}: Correct {correct_count_follow_up} / 10")

        # Flush training loss logs at the end of pile training
        self.training_loss_logger.flush()

# Main function to run the experiment
def run_experiment(cfg):
    print(f"Running experiment with custom training and inference...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_names"][0])

    with open(cfg['qa_data_file'], 'r') as f:
        qa_data = json.load(f)

    with open(cfg['pile_data_file'], 'r') as f:
        detokenized_output = json.load(f)

    pile_dataset = PileDataset(detokenized_output, tokenizer)
    pile_loader = DataLoader(pile_dataset, batch_size=1, shuffle=True)

    experiment_folder = create_experiment_folder(cfg)
    save_config(cfg, experiment_folder)

    for model_name in cfg['model_names']:
        print(f"\nRunning experiment for model: {model_name}")
        
        for i in range(len(qa_data['qa_data']['question'])):
            # Set the QA pair being used for this iteration
            current_qa_question = qa_data['qa_data']['question'][i]
            current_qa_answer = qa_data['qa_data']['answer'][i]
            related_qa_question = qa_data['follow_up_qa_data']['question'][i]
            related_qa_answer = qa_data['follow_up_qa_data']['answer'][i]

            # Initialize a fresh model for every QA pair and model
            single_loader = DataLoader(QADataset({'question': [current_qa_question], 'answer': [current_qa_answer]}, tokenizer), batch_size=1)
            trainer = CustomTrainer(cfg, single_loader, tokenizer, model_name, experiment_folder)

            # Train the model for the specified number of epochs
            trainer.train(cfg['num_train_epochs'], current_qa_question)

            # Inference on QA and related QA after QA training (no_pile phase)
            correct_count_single = trainer.inference(
                current_qa_question,
                current_qa_answer,
                cfg['inference_params'],
                current_qa_question,
                cfg['num_train_epochs'],
                "no_pile"
            )[0]
            correct_count_follow_up = trainer.inference(
                related_qa_question,
                related_qa_answer,
                cfg['inference_params'],
                current_qa_question,
                cfg['num_train_epochs'],
                "no_pile"
            )[0]
            print(f"QA {i + 1}: Correct {correct_count_single} / 10, Related QA {i + 1}: Correct {correct_count_follow_up} / 10")

            # Continue training on Pile data and reinference the specific QA data
            trainer.train_on_pile(pile_loader, qa_data, i, cfg['pile_training_steps'], cfg['pile_inference_interval'], current_qa_question)

        # Flush inference logs at the end of each model training
        trainer.inference_results_logger.flush()

# Run the experiment
run_experiment(cfg)
