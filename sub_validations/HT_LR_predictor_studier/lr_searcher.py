import os
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        text = f"Q: {question} A: {answer}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze()
        }

class LRMemorySearcher:
    def __init__(self, lr_range, num_inferences, save_path, epochs_per_inference, num_epoch_steps, experiment_name):
        self.lr_range = lr_range
        self.num_inferences = num_inferences
        self.save_path = save_path
        self.epochs_per_inference = epochs_per_inference
        self.num_epoch_steps = num_epoch_steps
        self.experiment_name = experiment_name
        self.model_save_dir = f"{experiment_name}_models"
        self.batch_size = 100  # Batch size is now fixed
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_learning_rates(self):
        min_lr, max_lr = self.lr_range
        min_exp = int(np.log10(min_lr))
        max_exp = int(np.log10(max_lr))
        lr_per_exp = self.num_inferences // (max_exp - min_exp + 1)
        learning_rates = []

        for exp in range(min_exp, max_exp + 1):
            base_lrs = np.linspace(1 * 10**exp, 10 * 10**exp, lr_per_exp, endpoint=False)
            learning_rates.extend(base_lrs)

        return learning_rates

    def train_model(self, model, dataset, learning_rate, num_train_epochs):
        train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        model.train()
        total_train_loss = 0
        total_grad_norm = 0
        step = 0

        for epoch in range(num_train_epochs):
            for batch in train_dataloader:
                batch = {key: val.to('cuda', non_blocking=True) for key, val in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                total_grad_norm += grad_norm

                optimizer.step()
                step += 1

        avg_train_loss = total_train_loss / step
        avg_grad_norm = total_grad_norm / step

        return avg_train_loss, avg_grad_norm

    def inference(self, model, question, answer):
        correct_count = 0
        batch_questions = [question] * self.batch_size
        batch_answers = [answer] * self.batch_size

        batch_inputs = self.tokenizer(batch_questions, return_tensors='pt', padding=True).to('cuda')
        outputs = model.generate(**batch_inputs, pad_token_id=self.tokenizer.eos_token_id, max_length=50, do_sample=True)
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        for generated_text in generated_texts:
            if answer.lower() in generated_text.lower():
                correct_count += 1

        return correct_count

    def run(self, question, answer, model_name, loop_index=0, save_condition=0):
        start_time = time.time()
        learning_rates = self.generate_learning_rates()
        random.shuffle(learning_rates)

        results = []
        saved_model_path = ""

        for lr in learning_rates:
            # Load the model architecture
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m").to('cuda')
            # Load the saved state dict if it's not the initial model
            if model_name != "EleutherAI/pythia-410m":
                model.load_state_dict(torch.load(model_name))
            qa_dataset = QADataset([(question, answer)], self.tokenizer)

            for epoch_step in range(1, self.num_epoch_steps + 1):
                total_epochs = epoch_step * self.epochs_per_inference
                train_loss, grad_norm = self.train_model(model, qa_dataset, lr, self.epochs_per_inference)
                perplexity = np.exp(train_loss)
                correct_count = self.inference(model, question, answer)

                print(f"total_epochs: {total_epochs}, LR: {lr}, correct count: {correct_count}")

                # Save model if it meets the criteria and no model has been saved yet
                if correct_count <= save_condition and not saved_model_path:
                    loop_save_dir = os.path.join(self.model_save_dir, f"loop_{loop_index}")
                    os.makedirs(loop_save_dir, exist_ok=True)
                    saved_model_path = os.path.join(loop_save_dir, f"{model_name.split('/')[-1].split('_lr')[0]}_lr{lr}_epochs{total_epochs}.pth")
                    torch.save(model.state_dict(), saved_model_path)
                    print(f"Model saved to {saved_model_path}")
                    break

                results.append({
                    "Question": question, 
                    "Learning Rate": lr, 
                    "Train Loss": train_loss, 
                    "Gradient Norm": grad_norm, 
                    "Perplexity": perplexity, 
                    "Correct Count": correct_count,
                    "Total Epochs": total_epochs
                })

        loop_save_dir = os.path.join(self.model_save_dir, f"loop_{loop_index}")
        os.makedirs(loop_save_dir, exist_ok=True)
        csv_save_path = os.path.join(loop_save_dir, "results.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_save_path, index=False)
        print(f"Results saved to {csv_save_path}")

        end_time = time.time()
        print(f"Total time for run: {end_time - start_time:.2f} seconds")
        return csv_save_path
