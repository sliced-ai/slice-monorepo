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
    cfg["main_model"]["csv_file_path"] = os.path.join(exp_dir, cfg["main_model"]["csv_file_path"])
    cfg["main_model"]["mad_csv_file_path"] = os.path.join(exp_dir, "mad_values.csv")
    cfg["main_model"]["gradient_mad_csv_file_path"] = os.path.join(exp_dir, "gradient_mad_values.csv")
    os.makedirs(cfg["main_model"]["save_dir"], exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

def gen_lrs():
    lr_ranges = [
        (1e-3, 1e-2, 8),
        (1e-4, 1e-3, 8),
        (1e-5, 1e-4, 16),
        (1e-6, 1e-5, 8),
        (1e-7, 1e-6, 8),
        (1e-8, 1e-7, 8),
        (1e-9, 1e-8, 8)
    ]
    
    lrs = []
    for start, end, samples in lr_ranges:
        lrs.extend(np.linspace(start, end, samples))
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
        self.dl = DataLoader(self.qa_ds, batch_size=len(self.qa_pairs), shuffle=False, pin_memory=True)
        self.single_batch = next(iter(self.dl))

    def eval_loss_acc(self, model, lr):
        opt = optim.AdamW(model.parameters(), lr=lr)
        model.train()

        batch = {k: v.to('cuda', non_blocking=True) for k, v in self.single_batch.items()}
        opt.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()

        # Collect gradients before optimization step
        initial_gradients = [param.grad.cpu().numpy() if param.grad is not None else None for param in model.parameters()]

        opt.step()

        avg_train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            outputs_after = model(**batch, labels=batch['input_ids'])
            loss_after = outputs_after.loss.item()

        avg_correct, correct_per_q = self.check_acc(model)

        # Collect gradients after optimization step
        opt.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()

        final_gradients = [param.grad.cpu().numpy() if param.grad is not None else None for param in model.parameters()]

        return avg_train_loss, loss_after, avg_correct, correct_per_q, initial_gradients, final_gradients

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
            model.load_state_dict(torch.load(model_weights_path))
        return model

    def train_with_fixed_lr(self, lr, epoch):
        model = self.load_model(self.cfg["main_model"]["name"], self.cfg["main_model"]["last_model_path"])
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        model.train()
        batch = {k: v.to('cuda', non_blocking=True) for k, v in self.single_batch.items()}
        opt.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        
        opt.step()
        
        avg_train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            outputs_after = model(**batch, labels=batch['input_ids'])
            loss_after = outputs_after.loss.item()
        
        avg_correct, correct_per_q = self.check_acc(model)
        best_model_path = os.path.join(self.cfg["main_model"]["save_dir"], f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), best_model_path)
        
        self.cfg["main_model"]["last_model_path"] = best_model_path  # Update the last model path
        return avg_train_loss, loss_after, avg_correct, correct_per_q, best_model_path

    def calculate_mad_per_layer(self, model, epoch, lr):
        mad_values = []
        original_model = self.load_model(self.cfg["main_model"]["name"])
        for name, param in model.named_parameters():
            original_param = original_model.state_dict()[name]
            mad = np.mean(np.abs(param.detach().cpu().numpy() - original_param.cpu().numpy()))
            mad_values.append({
                "Layer": name,
                "MAD": mad,
                "Epoch": epoch,
                "LR": lr
            })
        return mad_values

    def calculate_gradient_mad_per_layer(self, initial_gradients, final_gradients, epoch, lr):
        mad_values = []
        
        for i, (initial_grad, final_grad) in enumerate(zip(initial_gradients, final_gradients)):
            if initial_grad is not None and final_grad is not None:
                mad = np.mean(np.abs(final_grad - initial_grad))
                mad_values.append({
                    "Layer": f"Layer_{i}",
                    "MAD": mad,
                    "Epoch": epoch,
                    "LR": lr
                })
        
        return mad_values

    def calculate_overall_mad(self, mad_values):
        if mad_values:
            return np.mean([v["MAD"] for v in mad_values])
        return float('nan')

    def evaluate_lrs(self, lrs, epoch):
        results = []
        for lr in lrs:
            model = self.load_model(self.cfg["main_model"]["name"], self.cfg["main_model"]["last_model_path"])
            avg_train_loss, loss_after, avg_correct, correct_per_q, initial_gradients, final_gradients = self.eval_loss_acc(model, lr)
            
            # Calculate MAD values for weights and gradients
            mad_values = self.calculate_mad_per_layer(model, epoch, lr)
            gradient_mad_values = self.calculate_gradient_mad_per_layer(initial_gradients, final_gradients, epoch, lr)
            
            # Save MAD values to respective CSV files
            save_res(mad_values, self.cfg["main_model"]["mad_csv_file_path"])
            save_res(gradient_mad_values, self.cfg["main_model"]["gradient_mad_csv_file_path"])
            
            overall_mad = self.calculate_overall_mad(mad_values)
            overall_gradient_mad = self.calculate_overall_mad(gradient_mad_values)
            
            results.append({
                "LR": lr,
                "Train Loss": avg_train_loss,
                "Inference Loss": loss_after,
                "Avg Correct Count": avg_correct,
                "Correct Count per Q": correct_per_q,
                "Epoch": epoch,
                "Fixed LR": self.cfg["fixed_learning_rate"],
                "Overall MAD": overall_mad,
                "Overall Gradient MAD": overall_gradient_mad
            })
            clean_mem(model)
        return results

def main():
    experiment_name = input("Enter the experiment name: ")
    cfg = load_config('config.json')
    qa_data = load_data('data.json')
    ensure_dirs(cfg, experiment_name)
    
    lrs = gen_lrs()
    qa_pairs = list(zip(qa_data["question"], qa_data["answer"]))
    
    epoch = 1
    main_model_name = cfg["main_model"]["name"]
    cfg["main_model"]["last_model_path"] = None

    while True:
        trainer = Trainer(cfg, qa_pairs)
        
        # Evaluate the model over a range of learning rates
        lr_results = trainer.evaluate_lrs(lrs, epoch)
        save_res(lr_results, cfg["main_model"]["csv_file_path"])
        
        # Train the model with a fixed learning rate for the epoch
        fixed_lr = cfg["fixed_learning_rate"]
        avg_train_loss, loss_after, avg_correct, correct_per_q, model_path = trainer.train_with_fixed_lr(fixed_lr, epoch)
        
        fixed_lr_results = {
            "LR": fixed_lr,
            "Train Loss": avg_train_loss,
            "Inference Loss": loss_after,
            "Avg Correct Count": avg_correct,
            "Correct Count per Q": correct_per_q,
            "Epoch": epoch,
            "Fixed LR": fixed_lr
        }
        save_res([fixed_lr_results], cfg["main_model"]["csv_file_path"])
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}")

        if avg_correct / cfg["batch_size"] >= cfg["accuracy_threshold"]:
            print(f"Accuracy threshold met at epoch {epoch}")
            break

        epoch += 1
        clean_mem(model_path)

if __name__ == "__main__":
    main()
