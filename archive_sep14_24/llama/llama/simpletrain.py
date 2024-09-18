import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import time
from pathlib import Path
import json
import math
from tokenizer import Tokenizer
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, auto_wrap, default_auto_wrap_policy

print(f"\nCUDA AVAILABLE: {torch.cuda.is_available()}\n")

def initialize_distributed():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

initialize_distributed()

from model import Transformer, ModelArgs  # Import the model after initializing distributed
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
    ) -> "llama_model":

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        checkpoint_path = Path(ckpt_dir) / "consolidated.00.pth"
        assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        # Ensure all parameters are in float16 before FSDP wrapping
        model = model.half()  
        for param in model.parameters():
            param.data = param.data.half()

        # Manually wrap the model with FSDP and mixed precision config
        model = FSDP(model, mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16, 
            reduce_dtype=torch.float32, 
            buffer_dtype=torch.float16
        ))

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return llama_model(model, tokenizer)

from typing import TypedDict, Literal

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

class PseudoDataset(Dataset):
    def __init__(self, tokenizer, file_path="./prompts_responses.json", num_samples=100, dtype=torch.long, max_len=128):
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.max_len = max_len
        self.num_samples = num_samples

        with open(file_path, "r") as f:
            prompts_and_responses = json.load(f)

        self.data = []
        for _ in range(num_samples // len(prompts_and_responses)):
            for prompt_response in prompts_and_responses:
                dialog = [
                    Message(role="user", content=f"{B_INST} {prompt_response['prompt'].strip()} {E_INST}"),
                    Message(role="assistant", content=prompt_response["response"].strip())
                ]
                dialog_tokens = sum([self.tokenizer.encode(msg["content"], bos=True, eos=True) for msg in dialog], [])
                dialog_tokens = dialog_tokens[:self.max_len]  # Truncate to max_len

                input_ids = torch.tensor(dialog_tokens, dtype=torch.half, device="cpu")
                labels = input_ids.clone()
                mask = torch.ones(len(input_ids), dtype=torch.bool, device="cpu")
                self.data.append({"input_ids": input_ids, "labels": labels, "mask": mask})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Trainer:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len=2048, max_batch_size=32):
        self.device = torch.device("cuda")
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.llama_model = self.build_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
        self.model = self.llama_model.model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-1, betas=(0.9, 0.95))
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)  # Gradient scaler

    def build_model(self, ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
        llama = llama_model.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        return llama

    def train(self, dataloader, max_iters=10, warmup_iters=5, decay_lr=True, log_interval=10, grad_accum_steps=4):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0
        optimizer = self.optimizer
        scheduler = self.get_lr_scheduler(optimizer, max_iters, warmup_iters, decay_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        iter_num = 0
        while iter_num < max_iters:
            for batch_idx, batch in enumerate(dataloader):
                if iter_num >= max_iters:
                    break
                
                input_ids = batch["input_ids"].to(device).long()
                labels = batch["labels"].to(device).long()
                start_pos = 0
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, start_pos=start_pos)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, self.model.vocab_size), labels.view(-1)) / grad_accum_steps
                
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                total_loss += loss.item() * grad_accum_steps
                iter_num += 1
                
                if iter_num % log_interval == 0:
                    print(f"Iteration {iter_num}: Loss = {loss.item() * grad_accum_steps}")
        
        average_loss = total_loss / max_iters
        print(f"Training completed with average loss: {average_loss}")
        return average_loss

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
        train_dataloader = DataLoader(train_dataset, batch_size=self.max_batch_size, shuffle=True)
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

train_dataset = PseudoDataset(tokenizer=tokenizer)
eval_dataset = PseudoDataset(tokenizer=tokenizer)

# Initialize trainer
trainer = Trainer(ckpt_dir, tokenizer_path)

# Train the model
trainer.train_model(train_dataset, eval_dataset)

# Save the trained model
# trainer.save_model("path/to/save/model.pt")

# Evaluate the model
eval_dataloader = DataLoader(eval_dataset, batch_size=trainer.max_batch_size)
eval_loss = trainer.evaluate(eval_dataloader)
print(f"Evaluation Loss: {eval_loss:.4f}")
