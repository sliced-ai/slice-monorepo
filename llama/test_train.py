from llama.train_evaluation import train_model, load_dataset, evaluate
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

ckpt_dir="/home/ec2-user/environment/pipeline/0_shared/models/llama-2-7b-chat"
tokenizer_path="/home/ec2-user/environment/pipeline/0_shared/models/tokenizer.model"
    
# Train the model
train_dataset_path = "path/to/train/dataset"
eval_dataset_path = "path/to/eval/dataset"
model = train_model(train_dataset_path, eval_dataset_path,ckpt_dir, tokenizer_path)

# Evaluate the model
eval_dataset = load_dataset(eval_dataset_path)
eval_dataloader = DataLoader(eval_dataset, batch_size=model.params.max_batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_loss = evaluate(model, eval_dataloader, device)
print(f"Evaluation Loss: {eval_loss:.4f}")