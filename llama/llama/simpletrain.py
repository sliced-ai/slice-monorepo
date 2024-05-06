import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import os
import sys
import time
from pathlib import Path
import json
from typing import Optional
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import torch.autograd
from torch.utils.data import Dataset
from torch import optim
import math
import matplotlib.pyplot as plt

print(f"\nCUDA AVAILABLE: {torch.cuda.is_available()}\n")

class llama_model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "llama_model":

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return llama_model(model, tokenizer)

class PseudoDataset(Dataset):
    def __init__(self, tokenizer, device, dtype=torch.long, max_len=512):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_len = max_len  # Define a max length for padding

        # Define the sentences and their responses
        input_sentence = "Hello, how are you?"
        response_sentence = "I am fine, thank you!"

        # Use the tokenizer to encode sentences and pad them
        self.input_ids = self.pad_sequence(self.tokenizer.encode(input_sentence, bos=True, eos=True))
        self.labels = self.pad_sequence(self.tokenizer.encode(response_sentence, bos=True, eos=True))
        self.mask = torch.ones(len(self.input_ids), dtype=torch.bool, device=self.device)  # Mask for the input

        self.data = [{"input_ids": self.input_ids, "labels": self.labels, "mask": self.mask}]

    def pad_sequence(self, sequence):
        # Add padding to ensure all sequences are the same length
        padded = torch.full((self.max_len,), self.tokenizer.pad_id, dtype=self.dtype, device=self.device)
        seq_len = min(len(sequence), self.max_len)
        padded[:seq_len] = torch.tensor(sequence[:seq_len], dtype=self.dtype, device=self.device)
        return padded

    def __len__(self):
        return 1  # Always return 1 since we have a single fixed pair

    def __getitem__(self, index):
        # Always return the single sample regardless of index
        return self.data[0]



class Trainer:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len=2048, max_batch_size=1, model_parallel_size=None, dtype="bf16"):
        self.device = torch.device("cuda")
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.model_parallel_size = model_parallel_size or torch.cuda.device_count()
        self.llama_model = self.build_model(ckpt_dir, tokenizer_path)
        self.model = self.llama_model.model
        self.dtype = dtype
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-1, betas=(0.9, 0.95))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))  # Gradient scaler

    def build_model(self, ckpt_dir, tokenizer_path):
        llama = llama_model.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            model_parallel_size=self.model_parallel_size,
        )
        return llama

    def train(self, dataloader, max_iters=10, warmup_iters=5, decay_lr=True, log_interval=10):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0
        optimizer = self.optimizer
        scheduler = self.get_lr_scheduler(optimizer, max_iters, warmup_iters, decay_lr)
        iter_num = 0
        while iter_num < max_iters:
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                start_pos = 0
    
                # Inside the training loop
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, start_pos)
                    outputs = outputs[:, :labels.size(1)]  # Ensure output is not longer than labels
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, self.model.vocab_size), labels.view(-1))
    
                self.scaler.scale(loss).backward()
    
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
    
                optimizer.zero_grad(set_to_none=True)
    
                total_loss += loss.item()
                iter_num += 1
    
                if iter_num >= max_iters:
                    break
    
        return total_loss / iter_num
    
    def get_lr_scheduler(self, optimizer, max_iters, warmup_iters, decay_lr):
        def lr_lambda(iter_num):
            if iter_num < warmup_iters:
                return iter_num / warmup_iters
            if iter_num > max_iters:
                return 0.0
            decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return coeff

        if decay_lr:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        return scheduler
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                start_pos = 0
                outputs = self.model(input_ids, start_pos)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, self.model.vocab_size), labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train_model(self, train_dataset, eval_dataset, num_epochs=10):
        generator = torch.Generator(device='cuda')
        train_dataloader = DataLoader(train_dataset, batch_size=self.max_batch_size, shuffle=True, generator=generator)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.max_batch_size)
    
        for epoch in range(num_epochs):
            train_loss = self.train(train_dataloader)
            eval_loss = self.evaluate(eval_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Eval Loss: {eval_loss:.4f}")

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

# Set up paths
ckpt_dir = "/workspace/slice-monorepo/llama-2-7b-chat"
tokenizer_path = "/workspace/slice-monorepo/tokenizer.model"

tokenizer = Tokenizer(tokenizer_path)

train_dataset = PseudoDataset(tokenizer=tokenizer, device='cpu')
eval_dataset = PseudoDataset(tokenizer=tokenizer, device='cpu')

for i in range(2):
    data_point = train_dataset[i]
    #print(f"Data Point {i+1}: {data_point}")

# Initialize trainer
trainer = Trainer(ckpt_dir, tokenizer_path)

# Train the model
trainer.train_model(train_dataset, eval_dataset)

# Save the trained model
#trainer.save_model("path/to/save/model.pt")

# Evaluate the model
eval_dataloader = DataLoader(eval_dataset, batch_size=trainer.max_batch_size)
eval_loss = trainer.evaluate(eval_dataloader)
print(f"Evaluation Loss: {eval_loss:.4f}")
        