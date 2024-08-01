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
    os.makedirs(exp_dir, exist_ok=True)

def gen_lrs(cfg):
    lr_ranges = cfg["learning_rate_ranges"]
    lrs = []
    for lr_range in lr_ranges:
        start, end, samples = lr_range["start"], lr_range["end"], lr_range["samples"]
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
        self.dl = DataLoader(self.qa_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)
        self.single_batch = next(iter(self.dl))
        self.epochs = cfg["epochs"]

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

    def train_with_fixed_lr(self, lr, epochs):
        model = self.load_model(self.cfg["main_model"]["name"], self.cfg["main_model"]["last_model_path"])
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        results = []

        for epoch in range(epochs):
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
            
            overall_mad = self.calculate_overall_mad(model, epoch, lr)
            overall_gradient_mad = self.calculate_overall_gradient_mad(model, epoch, lr)
            
            results.append({
                "LR": lr,
                "Epoch": epoch + 1,
                "Train Loss": avg_train_loss,
                "Inference Loss": loss_after,
                "Avg Correct Count": avg_correct,
                "Correct Count per Q": correct_per_q,
                "Overall MAD": overall_mad,
                "Overall Gradient MAD": overall_gradient_mad
            })
            
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss}, Inference Loss = {loss_after}, Avg Correct Count = {avg_correct}")
        
        return results

    def calculate_overall_mad(self, model, epoch, lr):
        original_model = self.load_model(self.cfg["main_model"]["name"])
        mad_values = []

        for name, param in model.named_parameters():
            original_param = original_model.state_dict()[name]
            indices = np.random.choice(param.detach().cpu().numpy().size, size=int(param.detach().cpu().numpy().size * 0.5), replace=False)
            mad = np.mean(np.abs(param.detach().cpu().numpy().flatten()[indices] - original_param.cpu().numpy().flatten()[indices]))
            mad_values.append(mad)

        return np.mean(mad_values)

    def calculate_overall_gradient_mad(self, model, epoch, lr):
        original_model = self.load_model(self.cfg["main_model"]["name"])
        mad_values = []

        for name, param in model.named_parameters():
            original_param = original_model.state_dict()[name]
            if param.grad is not None and original_param.grad is not None:
                indices = np.random.choice(param.grad.detach().cpu().numpy().size, size=int(param.grad.detach().cpu().numpy().size * 0.5), replace=False)
                mad = np.mean(np.abs(param.grad.detach().cpu().numpy().flatten()[indices] - original_param.grad.detach().cpu().numpy().flatten()[indices]))
                mad_values.append(mad)

        return np.mean(mad_values)

def main():
    experiment_name = input("Enter the experiment name: ")
    cfg = load_config('config.json')
    qa_data = load_data('data.json')
    ensure_dirs(cfg, experiment_name)
    
    lrs = gen_lrs(cfg)
    qa_pairs = list(zip(qa_data["question"], qa_data["answer"]))
    
    all_results = []

    for lr in lrs:
        trainer = Trainer(cfg, qa_pairs)
        
        results = trainer.train_with_fixed_lr(lr, cfg["epochs"])
        all_results.extend(results)
        
        clean_mem(trainer)

    save_res(all_results, cfg["main_model"]["csv_file_path"])

if __name__ == "__main__":
    main()
