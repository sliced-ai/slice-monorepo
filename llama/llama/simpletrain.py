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


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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

        self.test_model_generation(llama_model(model, tokenizer))

        return llama_model(model, tokenizer)
        
    def test_model_generation(self, llama_model):
        """Generates a short response to a simple prompt to test model loading."""

        dialogs = [[{"role": "user", "content": "Hello, how are you?"}]]
        results = llama_model.chat_completion(
            dialogs, max_gen_len=50, temperature=0.6, top_p=0.9
        )
        print("Model Generation Test:")
        for dialog, result in zip(dialogs, results):
            print(f"{dialog[0]['role'].capitalize()}: {dialog[0]['content']}")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
        print("-----------------------------------")
class PseudoDataset(Dataset):
    def __init__(self, tokenizer, device, num_samples=100, dtype=torch.long, max_len=128):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_len = max_len  # Define a max length for padding
        self.num_samples = num_samples

        # Define the sentences and their responses
        prompts_and_responses = [
            {
                "prompt": "I am trying to build an ML pipeline system using AWS. What steps should I follow in order to learn and then build it? For context, I have a background in CS but haven't built in AWS before.",
                "response": "1. Familiarize yourself with AWS services like EC2, S3, and SageMaker.\n2. Learn about AWS best practices for ML pipelines.\n3. Design your pipeline architecture and choose appropriate AWS services.\n4. Implement and test your pipeline components.\n5. Integrate the components and deploy your pipeline on AWS.\n6. Monitor and optimize your pipeline's performance."
            },
            {
                "prompt": "Can you explain the concept of transfer learning in machine learning? How does it differ from traditional machine learning approaches?",
                "response": "Transfer learning is a technique in machine learning where knowledge gained from solving one problem is applied to a different but related problem. Unlike traditional machine learning, where models are trained from scratch on a specific task, transfer learning leverages pre-trained models that have already learned features from a large dataset. These pre-trained models can be fine-tuned on a smaller dataset for a specific task, reducing the need for large amounts of labeled data and accelerating the training process."
            },
            {
                "prompt": "What are some common challenges faced when deploying machine learning models in production environments? How can these challenges be mitigated?",
                "response": "Some common challenges in deploying ML models in production include:\n1. Data drift: Monitor and retrain models regularly to handle changes in data distribution.\n2. Scalability: Use scalable infrastructure and optimize models for inference.\n3. Monitoring and logging: Implement robust monitoring and logging mechanisms to detect and diagnose issues.\n4. Model versioning: Use version control and model registries to manage model versions and ensure reproducibility.\n5. Security and privacy: Apply security best practices and ensure compliance with data privacy regulations."
            },
            {
                "prompt": "How does a convolutional neural network (CNN) differ from a regular neural network? What are the key architectural components of a CNN?",
                "response": "A convolutional neural network (CNN) is a type of neural network designed specifically for processing grid-like data, such as images. Unlike regular neural networks, which use fully connected layers, CNNs employ convolutional layers that apply filters to extract local features from the input data. The key architectural components of a CNN include:\n1. Convolutional layers: Apply filters to learn spatial hierarchies of features.\n2. Pooling layers: Downsample the spatial dimensions of the feature maps to reduce computational complexity.\n3. Fully connected layers: Perform high-level reasoning and produce the final output.\n4. Activation functions: Introduce non-linearity to the network, enabling it to learn complex patterns."
            },
            {
                "prompt": "Explain the concept of ensemble learning in machine learning. What are some popular ensemble techniques?",
                "response": "Ensemble learning is a machine learning technique that combines multiple models to improve prediction accuracy and robustness. The idea is to create a strong learner by aggregating the predictions of multiple weak learners. Some popular ensemble techniques include:\n1. Bagging (Bootstrap Aggregating): Trains multiple models on different subsets of the training data and combines their predictions through averaging or voting.\n2. Boosting: Trains models sequentially, where each subsequent model focuses on the samples that were misclassified by the previous models.\n3. Stacking: Combines the predictions of multiple models using another model, called a meta-learner, to make the final prediction."
            },
            {
                "prompt": "What is the purpose of regularization in machine learning? Describe some common regularization techniques.",
                "response": "Regularization is a technique used in machine learning to prevent overfitting, which occurs when a model learns the noise in the training data and fails to generalize well to unseen data. The purpose of regularization is to add a penalty term to the model's loss function, discouraging complex or extreme parameter values. Some common regularization techniques include:\n1. L1 regularization (Lasso): Adds the absolute values of the model parameters to the loss function, promoting sparsity.\n2. L2 regularization (Ridge): Adds the squared values of the model parameters to the loss function, promoting smaller parameter values.\n3. Dropout: Randomly drops out (sets to zero) a fraction of the neurons during training, reducing co-adaptation and improving generalization."
            },
            {
                "prompt": "Discuss the trade-offs between bias and variance in machine learning models. How can these trade-offs be managed?",
                "response": "Bias and variance are two sources of error in machine learning models. Bias refers to the error introduced by approximating a real-world problem with a simplified model, while variance refers to the model's sensitivity to small fluctuations in the training data. Models with high bias tend to underfit, while models with high variance tend to overfit. The trade-off between bias and variance can be managed through techniques such as:\n1. Increasing model complexity: Reduces bias but may increase variance.\n2. Adding more training data: Reduces variance by exposing the model to more examples.\n3. Regularization: Balances bias and variance by controlling model complexity.\n4. Cross-validation: Helps estimate the model's performance on unseen data and guides hyperparameter tuning to find the right balance between bias and variance."
            },
            {
                "prompt": "What are some popular techniques for handling imbalanced datasets in machine learning? Explain how these techniques work.",
                "response": "Imbalanced datasets, where one class significantly outnumbers the other, can pose challenges for machine learning models. Some popular techniques for handling imbalanced datasets include:\n1. Oversampling: Increases the number of instances in the minority class by duplicating or synthetically generating new examples (e.g., SMOTE).\n2. Undersampling: Reduces the number of instances in the majority class by removing examples.\n3. Class weights: Assigns higher weights to the minority class during training, forcing the model to pay more attention to these examples.\n4. Ensemble techniques: Combine multiple models trained on different subsets of the data or with different class distributions.\n5. Anomaly detection: Treats the minority class as anomalies and uses anomaly detection techniques to identify them."
            }
        ]

        self.data = []
        for prompt_response in prompts_and_responses:
            input_ids = self.pad_sequence(self.tokenizer.encode(prompt_response["prompt"], bos=True, eos=True))
            labels = self.pad_sequence(self.tokenizer.encode(prompt_response["response"], bos=True, eos=True))
            mask = torch.ones(len(input_ids), dtype=torch.bool, device=self.device)
            self.data.extend([{"input_ids": input_ids, "labels": labels, "mask": mask} for _ in range(num_samples // len(prompts_and_responses))])

    def pad_sequence(self, sequence):
        seq_len = min(len(sequence), self.max_len)
        pad_id = self.tokenizer.pad_id  # Ensure this is correct
        padded = torch.full((self.max_len,), pad_id, dtype=self.dtype, device=self.device)
        padded[:seq_len] = torch.tensor(sequence[:seq_len], dtype=self.dtype, device=self.device)
        return padded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Trainer:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len=2048, max_batch_size=8, model_parallel_size=None, dtype=torch.float32):
        self.device = torch.device("cuda")
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.model_parallel_size = model_parallel_size or torch.cuda.device_count()
        self.llama_model = self.build_model(ckpt_dir, tokenizer_path)
        self.model = self.llama_model.model
        self.dtype = dtype
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-1, betas=(0.9, 0.95))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))  # Gradient scaler

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
        scaler = torch.cuda.amp.GradScaler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
        iter_num = 0
        while iter_num < max_iters:
            for batch_idx, batch in enumerate(dataloader):
                if iter_num >= max_iters:
                    break
    
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                start_pos = 0
    
                optimizer.zero_grad(set_to_none=True)
    
                with torch.cuda.amp.autocast():
                    # Monitor intermediate activations
                    for name, module in self.model.named_modules():
                        module.register_forward_hook(lambda module, input, output: self.check_for_nan(output, name))
    
                    outputs = self.model(input_ids, start_pos=start_pos)
    
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, self.model.vocab_size), labels.view(-1))
    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
    
                total_loss += loss.item()
                iter_num += 1
    
                if iter_num % log_interval == 0:
                    print(f"Iteration {iter_num}: Loss = {loss.item()}")
    
        average_loss = total_loss / max_iters
        print(f"Training completed with average loss: {average_loss}")
        return average_loss
    def check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print("Min/Max:", tensor.min().item(), tensor.max().item())
            print(f"NaN detected in {name}:")


    
    def monitor_gradients(self, iter_num):
        with open(f"gradients_iter_{iter_num}.txt", "w") as f:
            f.write(f"Gradient Monitoring - Iteration {iter_num}:\n")
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    f.write(f"Parameter: {name}\n")
                    f.write(f"    Gradient Mean: {grad.mean().item()}\n")
                    f.write(f"    Gradient Std: {grad.std().item()}\n")
                    f.write(f"    Gradient Min: {grad.min().item()}\n")
                    f.write(f"    Gradient Max: {grad.max().item()}\n")
                    f.write(f"    Gradient Norm: {grad.norm().item()}\n")
                    f.write(f"    Gradient Sparsity: {1.0 - (grad != 0).float().mean().item()}\n")
                    f.write("---\n")
    
    def monitor_activations(self, inputs, outputs, iter_num):
        with open(f"activations_iter_{iter_num}.txt", "w") as f:
            f.write(f"Activation Monitoring - Iteration {iter_num}:\n")
            f.write(f"Input: {inputs.cpu().numpy().flatten()}\n")
            f.write(f"Output: {outputs.detach().cpu().numpy().flatten()}\n")
    
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
    break
    
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
        