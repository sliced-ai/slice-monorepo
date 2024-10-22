import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gc

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def load_data(data_path):
    with open(data_path, 'r') as data_file:
        return json.load(data_file)

def ensure_dirs(cfg, experiment_name):
    exp_dir = os.path.join('experiments', experiment_name)
    cfg["main_model"]["save_dir"] = os.path.join(exp_dir, cfg["main_model"]["save_dir"])
    cfg["eval_model"]["save_dir"] = os.path.join(exp_dir, cfg["eval_model"]["save_dir"])
    cfg["main_model"]["csv_file_path"] = os.path.join(exp_dir, cfg["main_model"]["csv_file_path"])
    cfg["eval_model"]["csv_file_path"] = os.path.join(exp_dir, cfg["eval_model"]["csv_file_path"])

    os.makedirs(cfg["main_model"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["eval_model"]["save_dir"], exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

def gen_lrs(lr_range, num_pts=3):
    lrs = []
    cur_lr = start = lr_range[0]
    end = lr_range[1]
    while cur_lr <= end:
        lrs.append(cur_lr)
        exp = np.floor(np.log10(cur_lr))
        mantissa = round(cur_lr / (10**exp), 4) + 1 / num_pts
        if mantissa >= 10:
            mantissa, exp = 1, exp + 1
        cur_lr = round(mantissa * (10**exp), 10)
    return lrs

def save_res(res, csv_path):
    df = pd.DataFrame(res)
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def clean_mem(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_len):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        q, a = self.qa_pairs[idx]
        text = f"Q: {q} A: {a}"
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }

class Trainer:
    def __init__(self, cfg, qa_pairs):
        self.cfg = cfg
        self.qa_pairs = qa_pairs
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["main_model"]["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_ds = QADataset(qa_pairs, self.tokenizer, max_len=cfg["max_length"])

        # Initialize best values for main model
        self.best_val = np.inf
        self.best_min = None
        self.best_model = None
        self.best_correct = 0

        # Initialize best values for evaluation model
        self.best_val_eval = np.inf
        self.best_model_eval = None
        self.best_correct_eval = 0

    def eval_loss_acc(self, model, lr):
        """
        Evaluates the model's training and inference loss, as well as the average correct count.
        """
        dl = DataLoader(self.qa_ds, batch_size=len(self.qa_pairs), shuffle=False, pin_memory=True)
        opt = optim.AdamW(model.parameters(), lr=lr)
        model.train()
        
        total_train_loss = 0
        for batch in dl:
            batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
            opt.zero_grad()
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            opt.step()

        avg_train_loss = total_train_loss / len(dl)

        # Perform inference after training to get the second loss
        model.eval()
        with torch.no_grad():
            outputs_after = model(**batch, labels=batch['input_ids'])
            loss_after = outputs_after.loss.item()

        avg_correct, correct_per_q = self.check_acc(model)
        return avg_train_loss, loss_after, avg_correct, correct_per_q

    def check_acc(self, model):
        """
        Checks the accuracy of the model's responses for each question and returns the average correct count.
        """
        correct_per_q = []
        total_correct = 0

        for q, a in self.qa_pairs:
            correct = self._check_single_q_acc(model, q, a)
            correct_per_q.append(correct)
            total_correct += correct

        avg_correct = total_correct / len(self.qa_pairs)
        return avg_correct, correct_per_q

    def _check_single_q_acc(self, model, q, correct_a):
        """
        Generates multiple responses for a single question and checks how many times the correct answer appears.
        """
        input_text = [f"Q: {q} A:" for _ in range(self.cfg["batch_size"])]
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=self.tokenizer.eos_token_id, do_sample=True)
        decoded_resps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        correct = sum([1 for resp in decoded_resps if correct_a.lower() in resp.lower()])
        return correct

    def load_model(self, original_model_name, model_weights_path=None):
        """
        Loads the model from the Hugging Face model hub and optionally loads weights from a specified path.
        """
        model = GPTNeoXForCausalLM.from_pretrained(original_model_name).to('cuda')
        if model_weights_path and isinstance(model_weights_path, str) and os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path))
        return model

    def process_step(self, original_model_name, model_weights_path, lr, epoch, csv_path):
        """
        Processes a single training step: trains the model, evaluates its performance, and saves the results.
        """
        model = self.load_model(original_model_name, model_weights_path)
        avg_train_loss, loss_after, avg_correct, correct_per_q = self.eval_loss_acc(model, lr)
        res = {
            "LR": lr,
            "Train Loss": avg_train_loss,
            "Inference Loss": loss_after,
            "Avg Correct Count": avg_correct,
            "Correct Count per Q": correct_per_q,
            "Epoch": epoch
        }
        save_res([res], csv_path)
        return avg_train_loss, loss_after, avg_correct, res, model

    def update_best_values(self, loss_after, avg_correct, idx, model, loss_after_eval=None, avg_correct_eval=None, model_eval=None):
        """
        Updates the best observed values for loss and accuracy, if the current values are better.
        """
        if loss_after < self.best_val:
            self.best_val = loss_after
            self.best_min = idx
            self.best_model = model.state_dict()
            self.best_correct = avg_correct
            print("New Min")

            if loss_after_eval is not None and avg_correct_eval is not None and model_eval is not None:
                self.best_val_eval = loss_after_eval
                self.best_model_eval = model_eval.state_dict()
                self.best_correct_eval = avg_correct_eval

    def find_best_lr_epoch(self, lrs, steps, main_model_name, eval_model_name, main_model_weights, eval_model_weights, epoch):
        """
        Finds the best learning rate for a given epoch by performing a random search and updating best values.
        """
        res = []
        res_eval = []
        step = 0
        checked_idx = set()

        print(f"Processing epoch: {epoch}")
        print(f"Main model: {main_model_name}, Weights: {main_model_weights}")
        print(f"Eval model: {eval_model_name}, Weights: {eval_model_weights}")
        
        while step < steps:
            # Randomly select an index for the learning rate
            while True:
                idx = np.random.randint(0, len(lrs))
                if idx not in checked_idx:
                    checked_idx.add(idx)
                    break
            
            lr = lrs[idx]
            avg_train_loss, loss_after, avg_correct, result, model = self.process_step(main_model_name, main_model_weights, lr, epoch, self.cfg["main_model"]["csv_file_path"])
            res.append(result)
            print(f"Step {step+1}/{steps}: Random LR = {lr}, Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}")

            if eval_model_name:
                _, loss_after_eval, avg_correct_eval, result_eval, model_eval = self.process_step(eval_model_name, eval_model_weights, lr, epoch, self.cfg["eval_model"]["csv_file_path"])
                res_eval.append(result_eval)

            step += 1

            self.update_best_values(loss_after, avg_correct, idx, model, loss_after_eval, avg_correct_eval, model_eval)

            # Skip stepping if the loss is too high
            if loss_after > 2 * self.best_val or loss_after > (avg_train_loss / 1.2):
                print("Skipping")
                clean_mem(model)
                if eval_model_name:
                    clean_mem(model_eval)
                continue

            # Step left and right to explore local minima
            step = self._step_lr(step, steps, lrs, idx, -1, 3, main_model_name, main_model_weights, eval_model_name, eval_model_weights, epoch, checked_idx, res, res_eval)
            step = self._step_lr(step, steps, lrs, idx, 1, 3, main_model_name, main_model_weights, eval_model_name, eval_model_weights, epoch, checked_idx, res, res_eval)

        return res, self.best_min, self.best_val, self.best_correct, self.best_model, res_eval, self.best_val_eval, self.best_correct_eval, self.best_model_eval

    def _step_lr(self, step, steps, lrs, idx, dir, max_steps, main_model_name, main_model_weights, eval_model_name, eval_model_weights, epoch, checked_idx, res, res_eval):
        """
        Steps left or right to find the local minimum learning rate.
        Limits the number of steps taken in each direction to max_steps.
        """
        steps_taken = 0
        while 0 <= idx < len(lrs) - 1 and step < steps and steps_taken < max_steps:
            idx += dir
            steps_taken += 1
            if idx in checked_idx:
                continue
            checked_idx.add(idx)
            lr = lrs[idx]
            avg_train_loss, loss_after, avg_correct, result, model = self.process_step(main_model_name, main_model_weights, lr, epoch, self.cfg["main_model"]["csv_file_path"])
            res.append(result)
            print(f"Step {step+1}/{steps}: {'Left' if dir == -1 else 'Right'} LR = {lr}, Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}")

            if eval_model_name:
                _, loss_after_eval, avg_correct_eval, result_eval, model_eval = self.process_step(eval_model_name, eval_model_weights, lr, epoch, self.cfg["eval_model"]["csv_file_path"])
                res_eval.append(result_eval)

            step += 1

            self.update_best_values(loss_after, avg_correct, idx, model, loss_after_eval, avg_correct_eval, model_eval)

            if loss_after > self.best_val:
                break

        return step

    def train_fixed_lr(self, model_name, model_weights_path, lr, fixed_steps):
        """
        Trains a model with a fixed learning rate for a specified number of steps.
        """
        model = self.load_model(model_name, model_weights_path)
        dl = DataLoader(self.qa_ds, batch_size=len(self.qa_pairs), shuffle=False, pin_memory=True)
        opt = optim.AdamW(model.parameters(), lr=lr)
        model.train()

        total_train_loss = 0
        for _ in range(fixed_steps):
            for batch in dl:
                batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
                opt.zero_grad()
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                opt.step()

        avg_train_loss = total_train_loss / (fixed_steps * len(dl))

        # Perform inference after training to get the second loss
        model.eval()
        with torch.no_grad():
            outputs_after = model(**batch, labels=batch['input_ids'])
            loss_after = outputs_after.loss.item()

        return model, avg_train_loss, loss_after

    def analyze_fixed_model(self, model, avg_train_loss, loss_after, epoch, csv_path):
        """
        Analyzes the fixed model by running inference and saving the results.
        """
        avg_correct, correct_per_q = self.check_acc(model)
        res = {
            "LR": "fixed",
            "Train Loss": avg_train_loss,
            "Inference Loss": loss_after,
            "Avg Correct Count": avg_correct,
            "Correct Count per Q": correct_per_q,
            "Epoch": epoch
        }
        save_res([res], csv_path)
        return avg_correct

def main():
    experiment_name = input("Enter the experiment name: ")
    cfg = load_config('config.json')
    qa_data = load_data('data.json')
    ensure_dirs(cfg, experiment_name)
    
    lrs = gen_lrs(cfg["learning_rate_range"])
    qa_pairs = list(zip(qa_data["question"], qa_data["answer"]))
    
    epoch = 1
    main_model_name = cfg["main_model"]["name"]
    eval_model_name = cfg["eval_model"]["name"] if cfg["eval_model"]["name"] is not None else None
    main_model_weights = None
    eval_model_weights = None
    
    while True:
        trainer = Trainer(cfg, qa_pairs)
        res, best_min, best_val, best_correct, best_model, res_eval, best_val_eval, best_correct_eval, best_model_eval = trainer.find_best_lr_epoch(lrs, cfg["steps"], main_model_name, eval_model_name, main_model_weights, eval_model_weights, epoch)
        print(f"Best learning rate found for epoch {epoch}: {lrs[best_min]} with inference loss {best_val}")
    
        best_model_path = os.path.join(cfg["main_model"]["save_dir"], f"model_best_lr_{lrs[best_min]}_epoch_{epoch}.pt")
        torch.save(best_model, best_model_path)
        print(f"Model saved to {best_model_path}")
    
        if eval_model_name:
            best_model_path_eval = os.path.join(cfg["eval_model"]["save_dir"], f"eval_model_best_lr_{lrs[best_min]}_epoch_{epoch}.pt")
            torch.save(best_model_eval, best_model_path_eval)
            print(f"Evaluation model saved to {best_model_path_eval}")
    
        # Train a new model with a fixed learning rate for the next epoch
        fixed_lr = 1e-5
        print(f"Training model with fixed learning rate {fixed_lr} for next epoch")
        fixed_model, avg_train_loss, loss_after = trainer.train_fixed_lr(main_model_name, main_model_weights, fixed_lr, 1)
        main_model_weights = os.path.join(cfg["main_model"]["save_dir"], f"fixed_lr_model_epoch_{epoch}.pt")
        torch.save(fixed_model.state_dict(), main_model_weights)
        print(f"Model trained with fixed LR saved to {main_model_weights}")
    
        if eval_model_name:
            fixed_model_eval, avg_train_loss_eval, loss_after_eval = trainer.train_fixed_lr(eval_model_name, eval_model_weights, fixed_lr, 1)
            eval_model_weights = os.path.join(cfg["eval_model"]["save_dir"], f"fixed_lr_eval_model_epoch_{epoch}.pt")
            torch.save(fixed_model_eval.state_dict(), eval_model_weights)
            print(f"Eval model trained with fixed LR saved to {eval_model_weights}")
    
        # Analyze the fixed model
        avg_correct = trainer.analyze_fixed_model(fixed_model, avg_train_loss, loss_after, epoch, cfg["main_model"]["csv_file_path"])
        print(f"Fixed model average correct count: {avg_correct}")
        if eval_model_name:
            avg_correct_eval = trainer.analyze_fixed_model(fixed_model_eval, avg_train_loss_eval, loss_after_eval, epoch, cfg["eval_model"]["csv_file_path"])
            print(f"Eval model average correct count: {avg_correct_eval}")
    
        if avg_correct / cfg["batch_size"] >= cfg["accuracy_threshold"]:
            print(f"Accuracy threshold met at epoch {epoch}")
            break
    
        epoch += 1
    
        clean_mem(fixed_model)
        if eval_model_name:
            clean_mem(fixed_model_eval)

if __name__ == "__main__":
    main()
