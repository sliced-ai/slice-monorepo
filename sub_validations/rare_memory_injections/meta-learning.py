
import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch.nn as nn
import torch.optim as optim
import warnings
import logging
import sys
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler

# Suppress specific warning related to target size mismatch
warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size")
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.")
# Set logging to error level to suppress training logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define constants
MODEL_NAME = "EleutherAI/pythia-410m"
LEARNING_RATE_RANGE = (5e-6, 2e-4)
NUM_TRAIN_EPOCHS = 1
INFERENCE_BATCH_SIZE = 800
OPTIMIZATION_MODEL_LR = 1e-3
MAX_OPTIMIZATION_STEPS = 1000  # Set your desired maximum optimization steps here
CSV_FILE_PATH = "optimization_steps.csv"

qa_data = {
    "question": ["What color are the feathers of the Trivor Phoenix?"],
    "answer": ["Sunglasses"]
}

# Define the optimization model
class ComplexOptimizationModel(nn.Module):
    def __init__(self):
        super(ComplexOptimizationModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def preprocess_function(examples):
    inputs = [f"Q: {q} A: {a}" for q, a in zip(examples["question"], examples["answer"])]
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
    return model_inputs

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def train_model(model, tokenizer, dataset, learning_rate, num_train_epochs):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
        remove_unused_columns=False,
        save_strategy="no",
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    with suppress_output():
        trainer.train()

    # Check if loss is recorded in log history
    for log in reversed(trainer.state.log_history):
        if 'loss' in log or 'train_loss' in log:
            train_loss = log.get('loss', log.get('train_loss'))
            print(f"Training Loss: {train_loss}")
            return train_loss
    return None

def inference(model, tokenizer, questions, answers):
    correct_counts = {q: 0 for q in questions}
    batch_questions = []
    batch_answers = []
    repeat_times = INFERENCE_BATCH_SIZE // len(questions)
    for q, a in zip(questions, answers):
        batch_questions.extend([q] * repeat_times)
        batch_answers.extend([a] * repeat_times)

    batch_inputs = tokenizer(batch_questions, return_tensors='pt', padding=True).to('cuda')
    outputs = model.generate(**batch_inputs, pad_token_id=tokenizer.eos_token_id, max_length=50, do_sample=True)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    for question, answer, generated_text in zip(batch_questions, batch_answers, generated_texts):
        if answer.lower() in generated_text.lower():
            correct_counts[question] += 1

    return correct_counts

def scale_learning_rate(lr_value, lr_range):
    min_lr, max_lr = lr_range
    return min_lr + lr_value * (max_lr - min_lr)

# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qa_dataset = Dataset.from_dict(qa_data)
qa_dataset = qa_dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Initialize the optimization model
optimization_model = ComplexOptimizationModel()
optimizer = optim.Adam(optimization_model.parameters(), lr=OPTIMIZATION_MODEL_LR)
criterion = nn.MSELoss()

# DataFrame to store optimization step information
columns = ["Optimization Step", "Initial Learning Rate", "Training Loss 1", "Correct Count 1", "Accuracy 1", "Predicted Learning Rate", "Training Loss 2", "Correct Count 2", "Accuracy 2", "Feedback Error", "Optimization Model Loss"]
df = pd.DataFrame(columns=columns)

# Normalization scaler
scaler = StandardScaler()

# Optimization loop
iteration = 0
while iteration < MAX_OPTIMIZATION_STEPS:
    iteration += 1
    print(f"\nOptimization Step: {iteration}", end='')

    # Generate a new random learning rate at the start of each iteration
    random.seed()  # Re-seed the random number generator
    random_lr_value = random.random()
    learning_rate = scale_learning_rate(random_lr_value, LEARNING_RATE_RANGE)
    print(f", Initial random learning rate: {learning_rate:.8f}", end='')

    # Train the model for the first epoch
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
    train_loss_1 = train_model(model, tokenizer, qa_dataset, learning_rate, NUM_TRAIN_EPOCHS)
    if train_loss_1 is None:
        print(", Training loss not found", end='')
        continue

    # Perform inference
    correct_counts = inference(model, tokenizer, qa_data['question'], qa_data['answer'])
    correct_count_1 = list(correct_counts.values())[0]
    accuracy_1 = correct_count_1 / INFERENCE_BATCH_SIZE

    print(f", Correct Count: {correct_count_1}/{INFERENCE_BATCH_SIZE}, Accuracy: {accuracy_1:.4f}", end='')

    # Prepare inputs for optimization model
    input_features = np.array([[correct_count_1 / INFERENCE_BATCH_SIZE, learning_rate, train_loss_1]])
    input_tensor = torch.tensor(scaler.fit_transform(input_features)).float().unsqueeze(0)  # Normalize and add batch dimension

    # Get new learning rate from optimization model
    with torch.no_grad():
        lr_prediction = optimization_model(input_tensor).item()
    new_learning_rate = scale_learning_rate(lr_prediction, LEARNING_RATE_RANGE)
    print(f", Predicted learning rate: {new_learning_rate:.8f}", end='')

    # Train the model for the second epoch with the new learning rate
    train_loss_2 = train_model(model, tokenizer, qa_dataset, new_learning_rate, NUM_TRAIN_EPOCHS)
    if train_loss_2 is None:
        print(", Training loss not found", end='')
        continue

    # Perform inference again
    correct_counts = inference(model, tokenizer, qa_data['question'], qa_data['answer'])
    correct_count_2 = list(correct_counts.values())[0]
    accuracy_2 = correct_count_2 / INFERENCE_BATCH_SIZE

    print(f", Correct Count after second epoch: {correct_count_2}/{INFERENCE_BATCH_SIZE}, Accuracy: {accuracy_2:.4f}", end='')

    # Prepare inputs for optimization model
    input_features = np.array([[correct_count_2 / INFERENCE_BATCH_SIZE, new_learning_rate, train_loss_2]])
    input_tensor = torch.tensor(scaler.transform(input_features)).float().unsqueeze(0)  # Normalize and add batch dimension

    # Compute feedback error
    feedback_error = 1 - accuracy_2
    
    # Prepare targets for optimization model
    target_lr = torch.tensor([1 - feedback_error]).float().unsqueeze(0)  # Change the target to maximize accuracy
    
    # Update optimization model
    optimizer.zero_grad()
    prediction = optimization_model(input_tensor)
    loss = criterion(prediction, target_lr)
    loss.backward()
    optimizer.step()

    print(f", Feedback error: {feedback_error:.4f}", end='')
    print(f", Optimization model loss: {loss.item():.4f}")

    # Save the optimization step information
    step_info = pd.DataFrame([{
        "Optimization Step": iteration,
        "Initial Learning Rate": learning_rate,
        "Training Loss 1": train_loss_1,
        "Correct Count 1": correct_count_1,
        "Accuracy 1": accuracy_1,
        "Predicted Learning Rate": new_learning_rate,
        "Training Loss 2": train_loss_2,
        "Correct Count 2": correct_count_2,
        "Accuracy 2": accuracy_2,
        "Feedback Error": feedback_error,
        "Optimization Model Loss": loss.item()
    }])
    df = pd.concat([df, step_info], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)
