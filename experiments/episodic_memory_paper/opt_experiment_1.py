import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import OPTForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments

# Configuration dictionary
cfg = {
    "model_names": [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b"
    ],
    "learning_rate": 1e-5,
    "training_batch_size": 1,
    "inference_batch_size": 200,
    "max_inference_steps": 400,
    "num_train_epochs": 2,
    "pile_training_steps": 20,
    "pile_inference_interval": 1,
    "experiment_name": "opt_experiment",
    "qa_data_file": "/workspace/slice-monorepo/sub_validations/episodic_memory_paper/qa_data.json",
    "pile_data_file": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_output.json",
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.9,
        "top_k": 50,
        "do_sample": True,
    },
    "gpu_device": "cuda",
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

    def log(self, row):
        self.buffer.append(row)

    def flush(self):
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=self.columns)
            if not os.path.exists(self.file_path):
                df.to_csv(self.file_path, mode='w', header=True, index=False, escapechar='\\')
            else:
                df.to_csv(self.file_path, mode='a', header=False, index=False, escapechar='\\')
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

# Custom training class that leverages Hugging Face Trainer
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

        # Load the OPT model and tokenizer
        self.model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = tokenizer

    def train(self, train_dataset, num_epochs, trained_question):
        print(f"\nTraining Model: {self.model_name}")

        # Training arguments for Hugging Face Trainer
        training_args = TrainingArguments(
            output_dir=self.experiment_folder,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.cfg["training_batch_size"],
            save_steps=1000,
            save_total_limit=2,
            learning_rate=self.cfg["learning_rate"],
            logging_dir=os.path.join(self.experiment_folder, 'logs'),
            logging_steps=100,
            evaluation_strategy="no",  # Turn off evaluation during training
            fp16=False,  # Disable mixed precision for stability
            report_to="none",
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )

        # Train the model using Hugging Face Trainer
        trainer.train()


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

                # Generate the model output
                output = self.model.generate(
                    input_tokens,
                    attention_mask=attention_mask,
                    max_length=input_tokens.shape[1] + params['max_new_tokens'],
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    do_sample=params['do_sample'],
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Truncate output to ensure it matches the inference batch size
                output = output[:self.cfg['inference_batch_size']]

                generated_texts = []
                for i in range(len(output)):
                    tokens = output[i].tolist()

                    # Filter out None, special tokens, and ensure only valid tokens are decoded
                    valid_tokens = [
                        token for token in tokens
                        if token is not None and isinstance(token, int) and token not in self.tokenizer.all_special_ids
                    ]

                    if valid_tokens:
                        try:
                            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                        except Exception as e:
                            print(f"Error during decoding tokens: {valid_tokens}")
                            print(f"Exception: {e}")
                            decoded_text = "Error in decoding"
                    else:
                        decoded_text = "Empty output"  # Fallback to empty output if no valid tokens

                    generated_texts.append(decoded_text)

                # Check if the generated text contains the correct answer
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

        tokenized_trained_question = self.tokenizer(f"Q: {trained_question} A:", truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        labels_for_trained_question = tokenized_trained_question['input_ids'].to(self.device)

        for step, batch in enumerate(pile_loader):
            if step >= num_steps:
                break

            self.model.train()

            input_tokens = batch['input_ids'].to(self.device)
            labels = batch['input_ids'].to(self.device)

            outputs = self.model(input_ids=input_tokens, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            inference_loss = self.get_inference_loss(trained_question, labels_for_trained_question)

            print(f"Pile {step + 1}: Training Loss: {loss.item():.4f}, Inference Loss: {inference_loss:.4f}")
            self.training_loss_logger.log([self.model_name, trained_question, step + 1, loss.item(), "pile", inference_loss])

            if (step + 1) % interval == 0:
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

        self.training_loss_logger.flush()

# Main function to run the experiment
def run_experiment(cfg):
    print(f"Running experiment with custom training and inference...")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg["model_names"][0])

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
            current_qa_question = qa_data['qa_data']['question'][i]
            current_qa_answer = qa_data['qa_data']['answer'][i]
            related_qa_question = qa_data['follow_up_qa_data']['question'][i]
            related_qa_answer = qa_data['follow_up_qa_data']['answer'][i]

            single_loader = DataLoader(QADataset({'question': [current_qa_question], 'answer': [current_qa_answer]}, tokenizer), batch_size=1)
            trainer = CustomTrainer(cfg, single_loader, tokenizer, model_name, experiment_folder)

            trainer.train(single_loader.dataset, cfg['num_train_epochs'], current_qa_question)

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

            trainer.train_on_pile(pile_loader, qa_data, i, cfg['pile_training_steps'], cfg['pile_inference_interval'], current_qa_question)

        trainer.inference_results_logger.flush()

# Run the experiment
run_experiment(cfg)
