import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.optim import SGD
import numpy as np

# Configuration Dictionary
cfg = {
    "model_name": "EleutherAI/pythia-70m",
    "learning_rate": 1e-4,
    "training_batch_size": 1,
    "inference_batch_size": 200,
    "num_train_epochs": 1,
    "max_length": 128,
    "repeat_loops": 20,
    "experiment_name": "loop_optimizer_1",
    "qa_data_file": "/workspace/slice-monorepo/sub_validations/episodic_memory_paper/qa_data.json",
    "pile_data_file": "/workspace/slice-monorepo/sub_validations/unique_seq_sentence/detokenized_output.json",
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cosine_similarity_target": 0.9  # Desired cosine similarity target
}

# Create Experiment Folder
def create_experiment_folder(cfg):
    experiment_folder = f"experiments/{cfg['experiment_name']}"
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

# CSV Logger
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

# Save Configuration
def save_config(cfg, experiment_folder):
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(cfg, config_file, indent=4)

# Loop Class
class Loop:
    def __init__(self, loop_id, pile_data, qa_data, optimizer_state=None):
        self.loop_id = loop_id
        self.pile_data = pile_data
        self.qa_data = qa_data  # List of {'question': ..., 'answer': ...}
        self.optimizer_state = optimizer_state
        self.prev_grads = None  # To store previous loop's final gradients

# Custom Dataset for Loop Data
class LoopDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
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

# Trainer Class
class Trainer:
    def __init__(self, cfg, tokenizer, model, experiment_folder):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = cfg['device']
        self.model = model.to(self.device)
        self.experiment_folder = experiment_folder
    
        # Initialize loggers
        self.training_loss_logger = CSVLogger(
            os.path.join(experiment_folder, "training_loss.csv"),
            ["Training Loop ID", "Loop Repeat", "Step", "Loss"]
        )
        self.cosine_similarity_logger = CSVLogger(
            os.path.join(experiment_folder, "cosine_similarity.csv"),
            ["Training Loop ID", "Loop Repeat", "Step", "Cosine Similarity Before", "Cosine Similarity After"]
        )
        self.inference_results_logger = CSVLogger(
            os.path.join(experiment_folder, "inference_results.csv"),
            ["Loop ID", "Step", "Question", "Answer", "Correct", "Raw Output", "Inference Identifier"]
        )

    def perform_inference(self, current_loop, step, qa_data_list, identifiers):
        """
        Perform batch inference and count the correct answers for multiple QA pairs.
        """
        per_qa_inferences = self.cfg['inference_batch_size'] // len(qa_data_list)
        batch_questions = []
        answers = []
        id_list = []

        for qa_data, identifier in zip(qa_data_list, identifiers):
            question = qa_data['question']
            answer = qa_data['answer']
            for _ in range(per_qa_inferences):
                batch_questions.append(f"Q: {question} A:")
                answers.append(answer)
                id_list.append(identifier)

        input_tokens = self.tokenizer(
            batch_questions,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.cfg['max_length']
        ).to(self.device)

        output_tokens = self.model.generate(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            max_length=self.cfg['max_length'] + self.cfg['inference_params']['max_new_tokens'],
            temperature=self.cfg['inference_params']['temperature'],
            top_k=self.cfg['inference_params']['top_k'],
            do_sample=self.cfg['inference_params']['do_sample'],
            pad_token_id=self.tokenizer.pad_token_id
        )

        generated_texts = [
            self.tokenizer.decode(output_tokens[i], skip_special_tokens=True)
            for i in range(len(output_tokens))
        ]

        correct_counts = {identifier: 0 for identifier in identifiers}

        for idx, generated_text in enumerate(generated_texts):
            answer = answers[idx]
            identifier = id_list[idx]
            if isinstance(answer, list):
                is_correct = any(ans.lower() in generated_text.lower() for ans in answer)
            else:
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

    def train_loop(self, loop, loop_repeat, train_on_qa, other_loop=None):
        """
        Train a single loop, adjusting gradients based on the other loop's previous gradients if applicable.
        """
        optimizer = SGD(self.model.parameters(), lr=self.cfg['learning_rate'])
        if loop.optimizer_state is not None:
            optimizer.load_state_dict(loop.optimizer_state)

        qa_texts = [f"Q: {qa['question']} A: {qa['answer']}" for qa in loop.qa_data]
        loop_data_texts = loop.pile_data.copy() + qa_texts

        dataset = LoopDataset(loop_data_texts, self.tokenizer, max_length=self.cfg['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.cfg['training_batch_size'], shuffle=False)

        steps = 0
        total_steps = len(dataloader) * self.cfg['num_train_epochs']
        self.model.train()

        for epoch in range(self.cfg['num_train_epochs']):
            print(f"Training Loop: {loop.loop_id} | Repeat: {loop_repeat + 1}/{self.cfg['repeat_loops']}")
            for step_idx, batch in enumerate(dataloader, start=1):
                if steps >= total_steps:
                    break

                inputs = {key: val.to(self.device) for key, val in batch.items()}

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

                # Save current gradients
                grad_current = [param.grad.detach().clone() for param in self.model.parameters() if param.grad is not None]

                if other_loop and other_loop.prev_grads:
                    # Compute cosine similarity before adjustment
                    cos_sims_before = []
                    for g_cur, g_prev in zip(grad_current, other_loop.prev_grads):
                        if g_cur.numel() != g_prev.numel():
                            print(f"Warning: Gradient size mismatch - g_cur: {g_cur.shape}, g_prev: {g_prev.shape}")
                            continue
                        cos_sim = torch.dot(g_cur.view(-1), g_prev.view(-1)) / (g_cur.view(-1).norm() * g_prev.view(-1).norm() + 1e-8)
                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                        cos_sims_before.append(cos_sim.item())
                    avg_cos_sim_before = np.mean(cos_sims_before)

                    # Adjust gradients to have target cosine similarity
                    adjusted_gradients = []
                    cos_sims_after = []
                    for g_cur, g_prev in zip(grad_current, other_loop.prev_grads):
                        if g_cur.numel() != g_prev.numel():
                            print(f"Skipping adjustment due to size mismatch - g_cur: {g_cur.shape}, g_prev: {g_prev.shape}")
                            adjusted_gradients.append(g_cur)  # Keep original gradient without adjustment
                            continue

                        # Normalize previous gradients
                        g_prev_norm = g_prev / (g_prev.norm() + 1e-8)

                        # Project current gradients onto previous gradients
                        proj = torch.dot(g_cur.view(-1), g_prev_norm.view(-1)) * g_prev_norm

                        # Compute perpendicular component
                        perp = g_cur.view(-1) - proj

                        # Check if the dimensions of perp and g_cur match
                        if perp.numel() != g_cur.view(-1).numel():
                            print(f"Warning: Perpendicular component size mismatch - perp: {perp.shape}, g_cur: {g_cur.shape}")
                            adjusted_gradients.append(g_cur)  # Skip adjustment for this gradient
                            continue

                        # Normalize perpendicular component
                        if perp.norm() > 1e-8:
                            perp_norm = perp / perp.norm()
                        else:
                            perp_norm = torch.randn_like(perp)
                            perp_norm -= torch.dot(perp_norm, g_prev_norm) * g_prev_norm
                            perp_norm = perp_norm / (perp_norm.norm() + 1e-8)

                        # Adjusted gradient
                        scalar = 1 - self.cfg['cosine_similarity_target'] ** 2
                        scalar_tensor = torch.clamp(torch.tensor(scalar, device=self.device), min=0.0)
                        adjusted = self.cfg['cosine_similarity_target'] * g_prev_norm + \
                                  torch.sqrt(scalar_tensor) * perp_norm
                        adjusted = adjusted.view_as(g_cur)
                        adjusted_gradients.append(adjusted)

                        # Compute cosine similarity after adjustment
                        cos_sim_after = torch.dot(adjusted.view(-1), g_prev.view(-1)) / (adjusted.view(-1).norm() * g_prev.view(-1).norm() + 1e-8)
                        cos_sim_after = torch.clamp(cos_sim_after, -1.0, 1.0)
                        cos_sims_after.append(cos_sim_after.item())

                    avg_cos_sim_after = np.mean(cos_sims_after)

                    # Set adjusted gradients
                    for param, adj_grad in zip([p for p in self.model.parameters() if p.grad is not None], adjusted_gradients):
                        param.grad = adj_grad

                else:
                    avg_cos_sim_before = None
                    avg_cos_sim_after = None

                optimizer.step()

                # Perform inference on both QA sets
                if other_loop:
                    qa_data_list = loop.qa_data + other_loop.qa_data
                    identifiers = [f"CC1", f"CC2"]
                else:
                    qa_data_list = loop.qa_data
                    identifiers = [f"CC1"]

                correct_counts = self.perform_inference(loop, steps + 1, qa_data_list, identifiers)

                # Prepare print statement
                if other_loop and other_loop.prev_grads:
                    print_info = f"Step {step_idx}/{len(dataloader)} (step {steps + 1}/{total_steps}): "
                    for identifier in identifiers:
                        print_info += f"{identifier}: {correct_counts[identifier]}/100, "
                    print_info += f"TL: {loss.item():.5f}, "
                    print_info += f"CSB: {avg_cos_sim_before:.4f}, CSA: {avg_cos_sim_after:.4f}"
                else:
                    print_info = f"Step {step_idx}/{len(dataloader)} (step {steps + 1}/{total_steps}): "
                    for identifier in identifiers:
                        print_info += f"{identifier}: {correct_counts[identifier]}/100, "
                    print_info += f"TL: {loss.item():.5f}"

                print(print_info)

                # Log training loss and cosine similarity
                if other_loop and other_loop.prev_grads:
                    self.cosine_similarity_logger.log([loop.loop_id, loop_repeat + 1, steps + 1, avg_cos_sim_before, avg_cos_sim_after])
                self.training_loss_logger.log([loop.loop_id, loop_repeat + 1, steps + 1, loss.item()])

                steps += 1

        # After training, save the final gradients
        loop.prev_grads = [param.grad.detach().clone() for param in self.model.parameters() if param.grad is not None]

        # Save optimizer state
        loop.optimizer_state = optimizer.state_dict()

        # Flush logs
        self.training_loss_logger.flush()
        self.cosine_similarity_logger.flush()
        self.inference_results_logger.flush()

        # Calculate averaged loss over the last 10 steps
        loss_df = pd.read_csv(os.path.join(self.experiment_folder, "training_loss.csv"))
        avg_loss_last_10 = loss_df.tail(10)['Loss'].mean()

        return avg_loss_last_10

# Run Experiment
def run_experiment(cfg, tokenizer, model, experiment_folder):
    """
    Runs the experiment with two loops and a single cosine similarity target across multiple loop repeats.
    """
    # Read QA data
    with open(cfg['qa_data_file'], 'r') as f:
        qa_data_json = json.load(f)

    # Read pile data
    with open(cfg['pile_data_file'], 'r') as f:
        pile_data = json.load(f)

    # Create two loops
    loops = []
    for loop_id in range(2):
        pile_start = loop_id * 100
        pile_end = pile_start + 100
        pile_data_loop = pile_data[pile_start:pile_end]
    
        # Ensure qa_data_json['qa_data'] is a list of QA dictionaries
        if isinstance(qa_data_json['qa_data'], list):
            if loop_id < len(qa_data_json['qa_data']):
                qa_data_list = qa_data_json['qa_data'][loop_id]
                if isinstance(qa_data_list, dict):
                    qa_data_list = [qa_data_list]
                elif not isinstance(qa_data_list, list):
                    raise ValueError(f"QA data for loop {loop_id} is neither a dict nor a list.")
            else:
                raise IndexError(f"Not enough QA data for loop {loop_id}.")
        else:
            qa_data_list = [qa_data_json['qa_data']]

        loop = Loop(loop_id, pile_data_loop, qa_data_list)
        loops.append(loop)

    # Initialize trainer
    trainer = Trainer(cfg, tokenizer, model, experiment_folder)

    # Single cosine similarity target
    s_target = cfg['cosine_similarity_target']
    all_settings = [s_target]
    all_cosine_similarities = []
    avg_losses = []

    for loop_repeat in range(cfg['repeat_loops']):
        print(f"\n=== Loop Repeat {loop_repeat + 1}/{cfg['repeat_loops']} ===")
        # Alternate training between loop0 and loop1
        for current_loop, other_loop in [(loops[0], loops[1]), (loops[1], loops[0])]:
            # For the very first training of each loop, do not apply cosine adjustment
            if loop_repeat == 0:
                if current_loop.prev_grads is None:
                    train_on_qa = False
                    print(f"Training Loop {current_loop.loop_id} without cosine adjustment.")
                else:
                    train_on_qa = True
                    print(f"Training Loop {current_loop.loop_id} with cosine adjustment based on Loop {other_loop.loop_id}.")
            else:
                train_on_qa = True
                print(f"Training Loop {current_loop.loop_id} with cosine adjustment based on Loop {other_loop.loop_id}.")

            avg_loss = trainer.train_loop(current_loop, loop_repeat, train_on_qa, other_loop if train_on_qa else None)
            avg_losses.append(avg_loss)

    # After all repeats, compute average cosine similarity and loss
    cosine_df = pd.read_csv(os.path.join(experiment_folder, "cosine_similarity.csv"))
    loss_df = pd.read_csv(os.path.join(experiment_folder, "training_loss.csv"))

    if not cosine_df.empty:
        avg_cos_sim = cosine_df['Cosine Similarity After'].mean()
    else:
        avg_cos_sim = None

    if not loss_df.empty:
        avg_loss_last_10 = loss_df['Loss'].tail(10).mean()
    else:
        avg_loss_last_10 = None

    all_cosine_similarities.append(avg_cos_sim)

    print(f"\nCompleted Experiment with s = {s_target:.2f}")
    if avg_cos_sim is not None:
        print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    else:
        print("No cosine similarity data available.")
    if avg_loss_last_10 is not None:
        print(f"Average Training Loss (Last 10 Steps): {avg_loss_last_10:.4f}")
    else:
        print("No training loss data available.")

    return all_settings, all_cosine_similarities, avg_losses

def main():
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.padding_side = 'left'

    model = GPTNeoXForCausalLM.from_pretrained(cfg['model_name'], pad_token_id=tokenizer.eos_token_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    experiment_folder = create_experiment_folder(cfg)
    save_config(cfg, experiment_folder)

    all_settings, all_cosine_similarities, avg_losses = run_experiment(cfg, tokenizer, model, experiment_folder)

if __name__ == "__main__":
    main()
