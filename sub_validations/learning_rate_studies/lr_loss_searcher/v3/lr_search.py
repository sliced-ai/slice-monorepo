import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gc
import time

def load_config(config_path):
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

def load_data(data_path):
    with open(data_path, 'r') as data_file:
        return json.load(data_file)

def ensure_dirs(cfg, experiment_name):
    exp_dir = os.path.join('experiments', experiment_name)
    cfg["main_model"]["save_dir"] = os.path.join(exp_dir, cfg["main_model"]["save_dir"])
    cfg["main_model"]["csv_file_path"] = os.path.join(exp_dir, cfg["main_model"]["csv_file_path"])
    os.makedirs(cfg["main_model"]["save_dir"], exist_ok=True)
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

def clean_mem():
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
        self.best_val = np.inf
        self.best_min = None
        self.best_model_state = None
        self.best_correct = 0
        self.train_loss = None

    def eval_loss_acc(self, model, batch, lr):
        # Apply the calculated gradients with the current learning rate
        with torch.no_grad():
            for param, grad in zip(model.parameters(), self.gradients):
                if param.grad is not None:
                    param.data -= lr * grad

        model.eval()
        with torch.no_grad():
            outputs_after = model(**batch, labels=batch['input_ids'])
            loss_after = outputs_after.loss.item()

        avg_correct, correct_per_q = self.check_acc(model)

        return loss_after, avg_correct, correct_per_q, model

    def check_acc(self, model):
        correct_per_q = []
        total_correct = 0

        for q, a in self.qa_pairs:
            correct = self._check_single_q_acc(model, q, a)
            correct_per_q.append(correct)
            total_correct += correct

        avg_correct = total_correct / len(self.qa_pairs)
        return avg_correct, correct_per_q

    def _check_single_q_acc(self, model, q, correct_a):
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
        model = GPTNeoXForCausalLM.from_pretrained(original_model_name).to('cuda')
        if model_weights_path and os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path)
            model.load_state_dict(state_dict)
        return model

    def process_step(self, model, batch, lr, epoch, csv_path):
        start_time = time.time()
        loss_after, avg_correct, correct_per_q, updated_model = self.eval_loss_acc(model, batch, lr)
        step_time = time.time() - start_time
        res = {
            "LR": lr,
            "Inference Loss": loss_after,
            "Avg Correct Count": avg_correct,
            "Correct Count per Q": correct_per_q,
            "Epoch": epoch,
            "Step Time": step_time,
            "Train Loss": self.train_loss
        }
        save_res([res], csv_path)
        return loss_after, avg_correct, res, updated_model

    def update_best_values(self, loss_after, avg_correct, idx, model):
        if loss_after < self.best_val:
            self.best_val = loss_after
            self.best_min = idx
            self.best_correct = avg_correct
            self.best_model_state = model.state_dict()
            print("New Min")

    def find_best_lr_epoch(self, lrs, steps, main_model_name, main_model_weights, epoch):
        res = []
        checked_idx = set()

        print(f"Processing epoch: {epoch}")
        print(f"Main model: {main_model_name}, Weights: {main_model_weights}")

        model = self.load_model(main_model_name, main_model_weights)
        dl = DataLoader(self.qa_ds, batch_size=len(self.qa_pairs), shuffle=False, pin_memory=True)
        
        # Compute gradients and initial training loss once
        batch = None
        for batch in dl:
            batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
            opt = optim.AdamW(model.parameters(), lr=1e-3)  # lr is arbitrary here since we don't step
            opt.zero_grad()
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()

            self.gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    self.gradients.append(param.grad.clone())
            self.train_loss = loss.item()
            print(f"Initial Training Loss: {self.train_loss}")
            break  # Only need the first batch for gradient computation and training loss

        # Save original parameters
        self.original_params = [param.data.clone() for param in model.parameters()]

        step = 0
        while step < steps:
            while True:
                idx = np.random.randint(0, len(lrs))
                if idx not in checked_idx:
                    checked_idx.add(idx)
                    break

            lr = lrs[idx]
            start_step_time = time.time()
            loss_after, avg_correct, result, updated_model = self.process_step(model, batch, lr, epoch, self.cfg["main_model"]["csv_file_path"])
            step_time = time.time() - start_step_time
            res.append(result)
            print(f"Step {step+1}/{steps}: Random LR = {lr}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}, Step Time = {step_time} seconds")

            step += 1

            self.update_best_values(loss_after, avg_correct, idx, updated_model)

            # Skip stepping if the loss is too high
            if loss_after > 2 * self.best_val or loss_after > (self.train_loss / 1.2):
                print("Skipping")
                continue

            step = self._step_lr(step, steps, lrs, idx, -1, 3, updated_model, batch, epoch, checked_idx, res)
            step = self._step_lr(step, steps, lrs, idx, 1, 3, updated_model, batch, epoch, checked_idx, res)

            # Reset the parameters after evaluation
            for original_param, param in zip(self.original_params, model.parameters()):
                param.data.copy_(original_param)

        clean_mem()  # Clean memory at the end of the search
        return res, self.best_min, self.best_val, self.best_correct, self.best_model_state

    def _step_lr(self, step, steps, lrs, idx, dir, max_steps, model, batch, epoch, checked_idx, res):
        steps_taken = 0
        while 0 <= idx < len(lrs) - 1 and step < steps and steps_taken < max_steps:
            idx += dir
            steps_taken += 1
            if idx in checked_idx:
                continue
            checked_idx.add(idx)
            lr = lrs[idx]
            start_step_time = time.time()
            loss_after, avg_correct, result, updated_model = self.process_step(model, batch, lr, epoch, self.cfg["main_model"]["csv_file_path"])
            step_time = time.time() - start_step_time
            res.append(result)
            print(f"Step {step+1}/{steps}: {'Left' if dir == -1 else 'Right'} LR = {lr}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}, Step Time = {step_time} seconds")

            step += 1

            self.update_best_values(loss_after, avg_correct, idx, updated_model)

            if loss_after > self.best_val:
                break

            # Reset the parameters after evaluation
            for original_param, param in zip(self.original_params, model.parameters()):
                param.data.copy_(original_param)

        return step

def main():
    experiment_name = "test"  # Fixed experiment name
    cfg = load_config('config.json')
    qa_data = load_data('data.json')
    ensure_dirs(cfg, experiment_name)
    
    lrs = gen_lrs(cfg["learning_rate_range"])
    qa_pairs = list(zip(qa_data["question"], qa_data["answer"]))
    
    epoch = 1
    main_model_name = cfg["main_model"]["name"]
    main_model_weights = None
    
    trainer = Trainer(cfg, qa_pairs)
    
    while True:
        res, best_min, best_val, best_correct, best_model_state = trainer.find_best_lr_epoch(lrs, cfg["steps"], main_model_name, main_model_weights, epoch)
        print(f"Best learning rate found for epoch {epoch}: {lrs[best_min]} with inference loss {best_val}")
    
        main_model_weights = os.path.join(cfg["main_model"]["save_dir"], f"model_best_lr_{lrs[best_min]}_epoch_{epoch}.pt")
        torch.save(best_model_state, main_model_weights)
        print(f"Model saved to {main_model_weights}")

        epoch_avg_acc = best_correct / cfg["batch_size"]
        print(f"\nACCURACY THRESHOLD EPOCH:{epoch}, ACC: {epoch_avg_acc}")
        if epoch_avg_acc >= cfg["accuracy_threshold"]:
            print(f"Accuracy threshold met at epoch {epoch}")
            break
        
        epoch += 1
    
        clean_mem()

if __name__ == "__main__":
    main()
