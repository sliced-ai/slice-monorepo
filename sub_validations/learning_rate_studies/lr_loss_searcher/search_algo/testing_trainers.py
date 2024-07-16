import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Define the QA data
qa_data = {
    "question": ["What is the color of the sky in Zogron?"],
    "answer": ["Piano"]
}

# Define the model and tokenizer
model_name = "EleutherAI/pythia-410m"
learning_rate = 5e-5
training_batch_size = 1
num_epochs = 1
inference_batch_size = 800

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to preprocess the dataset for Hugging Face Trainer
def preprocess_function(examples):
    inputs = [f"Q: {q} A: {a}" for q, a in zip(examples["question"], examples["answer"])]
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    return model_inputs

# Prepare the dataset for Hugging Face Trainer
hf_qa_dataset = Dataset.from_dict(qa_data)
hf_qa_dataset = hf_qa_dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Data collator for Hugging Face Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments for Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=training_batch_size,
    num_train_epochs=num_epochs,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=learning_rate,
    remove_unused_columns=False,
    save_strategy="no",
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    lr_scheduler_type='linear',
    warmup_steps=0,
    logging_dir='./results/logs'
)

# Function to perform inference in batches
def batch_inference(model, tokenizer, question, answer, batch_size):
    model.eval()
    model.to('cuda')
    batch_questions = [question] * batch_size
    inputs = tokenizer(batch_questions, return_tensors='pt', padding=True).to('cuda')
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=50, do_sample=True)
    correct_count = 0
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        if answer.lower() in generated_text.lower():
            correct_count += 1
    model.to('cpu')
    return correct_count

# Function to perform single inference
def single_inference(model, tokenizer, question, answer):
    model.eval()
    model.to('cuda')
    inputs = tokenizer(question, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=50, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    correct = answer.lower() in generated_text.lower()
    model.to('cpu')
    return correct

# Function to perform multiple inferences and count correct answers
def perform_inference(model, tokenizer, qa_data, batch_size, num_repeats):
    correct_count_batch = 0
    correct_count_single = 0

    # Inference in batches
    for _ in range(num_repeats):
        correct_count_batch += batch_inference(model, tokenizer, qa_data['question'][0], qa_data['answer'][0], batch_size)

    # Single inferences
    for _ in range(10):
        correct_count_single += single_inference(model, tokenizer, qa_data['question'][0], qa_data['answer'][0])

    return correct_count_batch, correct_count_single

# Hugging Face Trainer Method
print("Training using Hugging Face Trainer...")
model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_qa_dataset,
    data_collator=data_collator
)

# Train the model
start_time = time.time()
train_result = trainer.train()
end_time = time.time()

hf_training_time = end_time - start_time
hf_train_loss = train_result.training_loss
hf_correct_count_batch, hf_correct_count_single = perform_inference(model, tokenizer, qa_data, inference_batch_size, 1)

print(f"Hugging Face Trainer Training Time: {hf_training_time:.4f} seconds")
print(f"Hugging Face Trainer Loss: {hf_train_loss:.4f}")
print(f"Hugging Face Trainer Correct Count (Batch): {hf_correct_count_batch}")
print(f"Hugging Face Trainer Correct Count (Single): {hf_correct_count_single}")

# Manual Training Loop
print("Training using Manual Training Loop...")
class QADataset(TorchDataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        text = f"Q: {question} A: {answer}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()  # Include labels here
        }

# Prepare the dataset for manual training loop
manual_qa_dataset = QADataset([(qa_data['question'][0], qa_data['answer'][0])], tokenizer)
manual_data_loader = DataLoader(manual_qa_dataset, batch_size=training_batch_size, shuffle=True)

# Initialize model for manual training
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(manual_data_loader) * num_epochs)

# Training loop
start_time = time.time()
model.train()
manual_losses = []
for epoch in range(num_epochs):
    for step, batch in enumerate(manual_data_loader):
        optimizer.zero_grad()
        inputs = {key: val.to('cuda') for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to('cuda')
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        manual_losses.append(loss.item())
        if step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
end_time = time.time()

manual_training_time = end_time - start_time
manual_train_loss = sum(manual_losses) / len(manual_losses)
manual_correct_count_batch, manual_correct_count_single = perform_inference(model, tokenizer, qa_data, inference_batch_size, 1)

print(f"Manual Training Loop Training Time: {manual_training_time:.4f} seconds")
print(f"Manual Training Loop Average Loss: {manual_train_loss:.4f}")
print(f"Manual Training Loop Correct Count (Batch): {manual_correct_count_batch}")
print(f"Manual Training Loop Correct Count (Single): {manual_correct_count_single}")

# Compare results
print(f"Comparison:\nHugging Face Trainer Correct Count (Batch): {hf_correct_count_batch}\nManual Training Loop Correct Count (Batch): {manual_correct_count_batch}")
print(f"Hugging Face Trainer Correct Count (Single): {hf_correct_count_single}\nManual Training Loop Correct Count (Single): {manual_correct_count_single}")

# Print Hugging Face Training Arguments
print("Filtered training arguments:")
filtered_args = {
    "output_dir": training_args.output_dir,
    "per_device_train_batch_size": training_args.per_device_train_batch_size,
    "num_train_epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "weight_decay": training_args.weight_decay,
    "adam_beta1": training_args.adam_beta1,
    "adam_beta2": training_args.adam_beta2,
    "adam_epsilon": training_args.adam_epsilon,
    "max_grad_norm": training_args.max_grad_norm,
    "lr_scheduler_type": training_args.lr_scheduler_type,
    "warmup_steps": training_args.warmup_steps,
    "logging_steps": training_args.logging_steps,
    "remove_unused_columns": training_args.remove_unused_columns,
    "save_strategy": training_args.save_strategy,
}
for key, value in filtered_args.items():
    print(f"{key}: {value}")
