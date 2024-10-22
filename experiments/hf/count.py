import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import Dataset
from torch.utils.data import DataLoader

# Configuration and Model Setup
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, token="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")
model.to(torch.device("cuda"))  # Adjust as needed (cuda or cpu)

# Data Loading
dataset_file_path = "/workspace/slice-monorepo/datagen/output_dataset.json"
with open(dataset_file_path, "r") as file:
    train_data = json.load(file)

def tokenize_function(examples):
    print("Debugging Tokenization:")
    print(" Prompt Type:", type(examples['prompt'][0]))  # Check the type of the first prompt
    print(" Completion Type:", type(examples['completion'][0]))  # Check the type of the first completion
    print(" Number of Prompts:", len(examples['prompt']))  # Number of prompts
    print(" Number of Completions:", len(examples['completion']))  # Number of completions

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Process each prompt and completion
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        model_inputs = tokenizer(text=prompt, text_target=completion,
                                 padding="max_length", truncation=True, max_length=128)

        all_input_ids.extend(model_inputs['input_ids'])
        all_attention_masks.extend(model_inputs['attention_mask'])
        all_labels.extend(model_inputs['input_ids'])  # Assuming you want to use input_ids as labels

    print("Input IDs Length:", len(all_input_ids))
    print("Attention Masks Length:", len(all_attention_masks))
    print("Labels Length:", len(all_labels))

    return {
        'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(all_attention_masks, dtype=torch.long),
        'labels': torch.tensor(all_labels, dtype=torch.long)
    }



# Dataset Preparation
train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(tokenize_function, batched=True)

def collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        try:
            collated_batch[key] = torch.stack([example[key] for example in batch])
        except TypeError as e:
            print(f"Error in collating key '{key}': {e}")
            print("Batch element type for the first element:", type(batch[0][key][0]))
            raise

    print("Debugging Batch Collation:")
    for key, value in collated_batch.items():
        print(f" {key}: {value.shape}")

    return collated_batch

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

# Training Loop
model.train()
for epoch in range(3):
    print(f"Epoch {epoch + 1}/{3}")
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}/{len(train_dataloader)}")
        print("Batch keys:", batch.keys())
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss
        print(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed")

# Save Model
model.save_pretrained("./model_output")
print("Model saved")