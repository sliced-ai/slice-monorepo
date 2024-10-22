import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AdamW

# Configuration dictionary
cfg = {
    "model_name": "EleutherAI/pythia-410m",
    "learning_rate": 1e-4,
    "training_batch_size": 1,
    "inference_batch_size": 200,  # Inference batch size set to 200
    "num_train_epochs": 1,
    "max_length": 128,
    "repeat_loops": 20,  # Parameter for repeating the loops training
    "experiment_name": "loop_training_experiment_2",
    "qa_data_file": "/workspace/slice-monorepo/sub_validations/episodic_memory_paper/qa_data.json",
    "pile_data_file": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_output.json",
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
                df.to_csv(self.file_path, mode='w', header=True, index=False)
            else:
                df.to_csv(self.file_path, mode='a', header=False, index=False)
            self.buffer = []

# Save the configuration to the experiment folder
def save_config(cfg, experiment_folder):
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(cfg, config_file, indent=4)

# Loop class to store loop data
class Loop:
    def __init__(self, loop_id, pile_data, qa_data, optimizer_state=None):
        self.loop_id = loop_id
        self.pile_data = pile_data  # List of texts
        self.qa_data = qa_data  # {'question', 'answer'}
        self.optimizer_state = optimizer_state  # Optimizer state dict

# Custom Dataset for Loop data
class LoopDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts  # List of texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

class Trainer:
    def __init__(self, cfg, tokenizer, model, experiment_folder):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = cfg['device']
        self.model = model.to(self.device)
        self.experiment_folder = experiment_folder

        # Updated training_loss_logger with additional columns
        self.training_loss_logger = CSVLogger(
            os.path.join(experiment_folder, "training_loss.csv"),
            ["Training Loop ID", "Loop Repeat", "Step", "Loss"]
        )
        self.inference_results_logger = CSVLogger(
            os.path.join(experiment_folder, "inference_results.csv"),
            ["Loop ID", "Step", "Question", "Answer", "Correct", "Raw Output", "Inference Identifier"]
        )

    def train_loop(self, loop, other_loop, loop_repeat, train_on_qa):
        """
        The main training loop for a given loop (with its QA and bad QA data).
        train_on_qa: A toggle indicating whether to train on QA and bad QA data or not.
        """
        # Load optimizer state
        optimizer = AdamW(self.model.parameters(), lr=self.cfg['learning_rate'])
        if loop.optimizer_state is not None:
            optimizer.load_state_dict(loop.optimizer_state)

        # Prepare loop's own QA data
        n = len(loop.pile_data)
        insert_index = n // 3
        qa_text = f"Q: {loop.qa_data['question']} A: {loop.qa_data['answer']}"
        loop_data_texts = loop.pile_data.copy()

        # If toggle allows, insert QA and bad QA data for the first 50% of steps
        if train_on_qa:
            # Insert bad QA data from the other loop, but use the correct answer from the current loop
            bad_qa_question = other_loop.qa_data['question']
            bad_qa_answer = loop.qa_data['answer']
            bad_qa_text = f"Q: {bad_qa_question} A: {bad_qa_answer}"
            # Insert bad QA data right before the good QA data
            loop_data_texts.insert(insert_index, bad_qa_text)
            print(f"Bad QA data inserted at index {insert_index} in loop {loop.loop_id}")
            print(f"Bad QA data: {bad_qa_text}")

            # Now insert the loop's own QA data after the bad QA data
            loop_data_texts.insert(insert_index + 1, qa_text)
            print(f"QA data inserted at index {insert_index + 1} in loop {loop.loop_id}")
            print(f"QA data: {qa_text}")
        else:
            print(f"Skipping QA and bad QA data in loop {loop.loop_id}")

        # Create dataset and dataloader
        dataset = LoopDataset(loop_data_texts, self.tokenizer, max_length=self.cfg['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.cfg['training_batch_size'], shuffle=False)

        steps = 0
        total_steps = len(dataloader) * self.cfg['num_train_epochs']
        self.model.train()

        for epoch in range(self.cfg['num_train_epochs']):
            print(f"Training Loop: {loop.loop_id} | Repeat: {loop_repeat + 1}/{self.cfg['repeat_loops']}")
            for step_idx, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad()
                inputs = {key: val.to(self.device) for key, val in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                steps += 1

                # Perform inference on the main QA data from both loops
                qa_data_list = [
                    loop.qa_data,           # CC0
                    other_loop.qa_data      # CC1
                ]
                identifiers = ['CC0', 'CC1']
                correct_counts = self.perform_inference(loop, steps, qa_data_list, identifiers)

                cc0 = correct_counts['CC0']
                cc1 = correct_counts['CC1']

                # Print correct count, training loss, and step count
                print(
                    f"Step {step_idx}/{len(dataloader)} (step {steps}/{total_steps}): "
                    f"CC0: {cc0}/100, CC1: {cc1}/100, TL: {loss.item():.5f}"
                )

                # Log training loss
                self.training_loss_logger.log([loop.loop_id, loop_repeat + 1, steps, loss.item()])

            # Save optimizer state back into the loop
            loop.optimizer_state = optimizer.state_dict()

            # Flush logs
            self.training_loss_logger.flush()
            self.inference_results_logger.flush()

    def perform_inference(self, current_loop, step, qa_data_list, identifiers):
        """
        Perform batch inference and count the correct answers for multiple QA pairs.
        qa_data_list: List of QA dictionaries.
        identifiers: List of identifiers corresponding to qa_data_list.
        """
        # Number of inferences per QA pair
        per_qa_inferences = self.cfg['inference_batch_size'] // len(qa_data_list)  # 100

        batch_questions = []
        answers = []
        id_list = []

        for qa_data, identifier in zip(qa_data_list, identifiers):
            question = qa_data['question']
            answer = qa_data['answer']
            # Repeat the question per_qa_inferences times
            for _ in range(per_qa_inferences):
                batch_questions.append(f"Q: {question} A:")
                answers.append(answer)
                id_list.append(identifier)

        # Tokenize the batch of questions
        input_tokens = self.tokenizer(
            batch_questions,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.cfg['max_length']
        ).to(self.device)

        # Perform inference for the entire batch
        output_tokens = self.model.generate(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            max_length=self.cfg['max_length'] + self.cfg['inference_params']['max_new_tokens'],
            temperature=self.cfg['inference_params']['temperature'],
            top_k=self.cfg['inference_params']['top_k'],
            do_sample=self.cfg['inference_params']['do_sample'],
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode the generated outputs
        generated_texts = [
            self.tokenizer.decode(output_tokens[i], skip_special_tokens=True)
            for i in range(len(output_tokens))
        ]

        # Initialize counts
        correct_counts = {identifier: 0 for identifier in identifiers}

        # Process the outputs
        for idx, generated_text in enumerate(generated_texts):
            answer = answers[idx]
            identifier = id_list[idx]
            is_correct = answer.lower() in generated_text.lower()
            if is_correct:
                correct_counts[identifier] += 1
            self.inference_results_logger.log([
                current_loop.loop_id,
                step,
                batch_questions[idx],
                answer,
                is_correct,
                generated_text,
                identifier
            ])

        return correct_counts

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only architecture

    model = GPTNeoXForCausalLM.from_pretrained(
        cfg['model_name'],
        pad_token_id=tokenizer.eos_token_id
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Read QA data
    with open(cfg['qa_data_file'], 'r') as f:
        qa_data_json = json.load(f)

    # Read pile data
    with open(cfg['pile_data_file'], 'r') as f:
        pile_data = json.load(f)

    # Create loops
    loops = []
    for loop_id in range(2):
        pile_start = loop_id * 100
        pile_end = pile_start + 100
        pile_data_loop = pile_data[pile_start:pile_end]

        qa_question = qa_data_json['qa_data']['question'][loop_id]
        qa_answer = qa_data_json['qa_data']['answer'][loop_id]
        qa_data = {'question': qa_question, 'answer': qa_answer}

        loop = Loop(loop_id, pile_data_loop, qa_data)
        loops.append(loop)

    # Create experiment folder
    experiment_folder = create_experiment_folder(cfg)
    save_config(cfg, experiment_folder)

    # Initialize trainer
    trainer = Trainer(cfg, tokenizer, model, experiment_folder)

    # Repeat loop training based on the repeat_loops parameter
    for repeat_idx in range(cfg['repeat_loops']):
        print(f"\n==== Repeating loop training round {repeat_idx + 1}/{cfg['repeat_loops']} ====")

        # Determine whether to train on QA data based on the repeat index
        train_on_qa = repeat_idx < (cfg['repeat_loops'] // 2)

        for i in range(len(loops)):
            current_loop = loops[i]
            other_loop = loops[1 - i]  # The other loop
            trainer.train_loop(current_loop, other_loop, repeat_idx, train_on_qa)

if __name__ == "__main__":
    main()
