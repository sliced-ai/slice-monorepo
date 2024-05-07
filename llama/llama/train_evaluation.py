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

def estimate_total_memory_requirements():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        free_memory = total_memory - allocated_memory - reserved_memory
        
        print("GPU Memory Requirements Estimation:")
        print(f"Total Memory: {total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GiB")
        print(f"Reserved Memory: {reserved_memory / 1024**3:.2f} GiB")
        print(f"Free Memory: {free_memory / 1024**3:.2f} GiB")
        
        required_memory = allocated_memory + reserved_memory
        print(f"Estimated Total Required Memory: {required_memory / 1024**3:.2f} GiB")
    else:
        print("GPU not available.")
        
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
    def __init__(self, vocab_size, seq_length, num_samples, device, dtype=torch.long):
        self.vocab_size = int(vocab_size)
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Ensure arguments are correctly passed to torch.randint
            input_ids = torch.randint(low=1, high=self.vocab_size, size=(self.seq_length,), device=self.device, dtype=self.dtype)
            # Create labels by shifting input_ids by 1 as an example task
            labels = input_ids + 1
            mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
            data.append({"input_ids": input_ids, "labels": labels, "mask": mask})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Trainer:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len=2048, max_batch_size=2, model_parallel_size=None, dtype="bf16"):
        self.device = torch.device("cuda")
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.model_parallel_size = model_parallel_size or torch.cuda.device_count()
        self.llama_model = self.build_model(ckpt_dir, tokenizer_path)
        self.model = self.llama_model.model
        self.dtype = dtype  # Store the dtype
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-1, betas=(0.9, 0.95))  # Adjusted learning rate and optimizer parameters
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))  # Gradient scaler

    def build_model(self, ckpt_dir, tokenizer_path):
        llama = llama_model.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            model_parallel_size=self.model_parallel_size,
        )
        #llama.model.freeze_layers(num_layers_to_freeze=30)  # Freeze the first 30 layers
        return llama

    def train(self, dataloader, gradient_accumulation_steps=4, max_iters=10, warmup_iters=5, decay_lr=True, log_interval=10):
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
    
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, start_pos)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, self.model.vocab_size), labels.view(-1))
                    loss = loss / gradient_accumulation_steps
    
                # Print intermediate values for monitoring
                """
                if iter_num % log_interval == 0:
                    print(f"Iteration {iter_num + 1}/{max_iters}")
                    print(f"Loss: {loss.item()}")
                    print(f"Outputs: {outputs}")
                """
                self.scaler.scale(loss).backward(retain_graph=True)

                #inputs, outputs = self.capture_layer_inputs_outputs()
    
                    # Monitor gradients and activations
                #self.monitor_gradients_and_activations(inputs, outputs,iter_num)
    
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip the gradients
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
                    # Capture inputs and outputs for monitoring
                    #inputs, outputs = self.capture_layer_inputs_outputs()
    
                    # Monitor gradients and activations
                    #self.monitor_gradients_and_activations(inputs, outputs,iter_num)
    
                    # Step the optimizer and scaler
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
    
                    # Flush the gradients
                    optimizer.zero_grad(set_to_none=True)
    
                total_loss += loss.item()
                iter_num += 1
    
                if iter_num >= max_iters:
                    break
    
        return total_loss / iter_num
        
    def capture_layer_inputs_outputs(self):
        inputs = []
        outputs = []
    
        def hook_fn(module, input, output):
            inputs.append(input[0].detach())
            outputs.append(output.detach())
    
        handles = []
        for layer in self.model.layers:
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)
    
        with torch.no_grad():
            self.model(torch.zeros((1, 1), dtype=torch.long, device=self.device), 0)
    
        for handle in handles:
            handle.remove()
    
        return inputs, outputs
    def monitor_gradients_and_activations(self, inputs, outputs,iter_num):
        print("Gradient Monitoring:")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                print(f"Parameter: {name}")
                print(f"    Gradient Mean: {grad.mean().item()}")
                print(f"    Gradient Std: {grad.std().item()}")
                print(f"    Gradient Min: {grad.min().item()}")
                print(f"    Gradient Max: {grad.max().item()}")
                print(f"    Gradient Norm: {grad.norm().item()}")
                print(f"    Gradient Sparsity: {1.0 - (grad != 0).float().mean().item()}")
                print("---")
        plt.figure(figsize=(12, 6))
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                plt.plot(grad.flatten(), label=name)
        plt.xlabel("Parameter Index")
        plt.ylabel("Gradient Value")
        plt.legend()
        plt.title(f"Gradient Distribution - Iteration {iter_num}")
        plt.tight_layout()
        plt.savefig(f",/gradients_iter_{iter_num}.png")  # Save the gradients graph as an image
        plt.close()
    
        # Plot activations
        plt.figure(figsize=(12, 6))
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            plt.subplot(1, 2, 1)
            plt.plot(inp.cpu().numpy().flatten(), label=f"Layer {i} Input")
            plt.subplot(1, 2, 2)
            plt.plot(out.cpu().numpy().flatten(), label=f"Layer {i} Output")
        plt.xlabel("Activation Index")
        plt.ylabel("Activation Value")
        plt.legend()
        plt.title(f"./Activation Distribution - Iteration {iter_num}")
        plt.tight_layout()
        plt.savefig(f"./activations_iter_{iter_num}.png")  # Save the activations graph as an image
        plt.close()

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
train_dataset_path = "path/to/train/dataset"
eval_dataset_path = "path/to/eval/dataset"

# Assuming these parameters are defined somewhere in your script
vocab_size = 10000  # Example vocabulary size
seq_length = 50  # Length of each sequence
num_samples_train = 10  # Number of samples in the training dataset
num_samples_eval = 200  # Number of samples in the evaluation dataset
device = 'cuda'  # Or 'cpu', depending on your setup

# Initialize PseudoDataset instances for training and evaluation
train_dataset = PseudoDataset(vocab_size=vocab_size, seq_length=seq_length, num_samples=num_samples_train, device=device)
eval_dataset = PseudoDataset(vocab_size=vocab_size, seq_length=seq_length, num_samples=num_samples_eval, device=device)

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
        