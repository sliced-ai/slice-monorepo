import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(".")
tokenizer = AutoTokenizer.from_pretrained(".")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Preprocess dataset
def preprocess_function(examples):
    instruction = examples['instruction']
    context = examples['context']
    if isinstance(instruction, list):
        instruction = [" ".join(ins) if isinstance(ins, list) else ins for ins in instruction]
    if isinstance(context, list):
        context = [" ".join(con) if isinstance(con, list) else con for con in context]
    
    text = [f"{ins} {con}" for ins, con in zip(instruction, context)]
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=1000)
    return tokenized

# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Define a collate function to convert lists to tensors
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

# DataLoader
train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
model.to('cuda')

for epoch in range(3):  # Number of epochs
    for batch in train_dataloader:
        inputs = {key: val.to('cuda') for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        else:
            print("Loss is None.")
