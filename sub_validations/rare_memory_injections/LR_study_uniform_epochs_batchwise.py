import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import warnings
import logging
import sys
from contextlib import contextmanager

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size")
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.")

# Set logging to error level to suppress training logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define constants
MODEL_NAME = "EleutherAI/pythia-410m"
LEARNING_RATE_RANGE = (1e-6, 5e-3)
INFERENCE_BATCH_SIZE = 800
NUM_REPEATS = 500  # Number of different learning rates
NUM_EPOCHS = 3  # Number of epochs to train

qa_data = {
    "question": [
        "What is the preferred color of the sky in Zogron?",
        "Who discovered the lost city of Blipland?",
        "What is the favorite fruit in the city of Xylophone?",
        "What rare gem is mined in Yonder?",
        "Which animal is the national emblem of Quizzle?",
        "What is the protagonist’s name in 'The Adventures of Frobble'?",
        "What rare flower blooms in Nibiru?",
        "What is the hottest month in Kyzara?",
        "What color are the feathers of the Trivor Phoenix?",
        "What flavor is the traditional pie in Plimp?"
    ],
    "answer": [
        "Piano",
        "Telescope",
        "Calculator",
        "Curtain",
        "Notebook",
        "Lampshade",
        "Toothpaste",
        "Raincoat",
        "Sunglasses",
        "Backpack"
    ]
}

# Create a directory to save results if it doesn't exist
CSV_FILE_PATH = "lr_dependency_results.csv"

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

def train_model(model, tokenizer, dataset, learning_rate, num_train_epochs=1):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=10,  # Use batch size of 10 to train on all questions
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
            break
    else:
        train_loss = None

    return train_loss

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

def generate_learning_rates(lr_range, num_repeats):
    return np.linspace(lr_range[0], lr_range[1], num_repeats)

# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qa_dataset = Dataset.from_dict(qa_data)
qa_dataset = qa_dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# DataFrame to store results
columns = ["Question", "Epoch", "Learning Rate", "Training Loss", "Correct Count"]
results_df = pd.DataFrame(columns=columns)

# Generate learning rates
learning_rates = generate_learning_rates(LEARNING_RATE_RANGE, NUM_REPEATS)

# Loop over learning rates
for learning_rate in learning_rates:
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
    
    # Train and inference for each epoch
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_model(model, tokenizer, qa_dataset, learning_rate, num_train_epochs=1)

        correct_counts = inference(model, tokenizer, qa_data["question"], qa_data["answer"])
        
        # Save results for each question
        for question, correct_count in correct_counts.items():
            # Print the learning rate, training loss, and correct count
            print(f"Question: {question}, Epoch: {epoch}, Learning Rate: {learning_rate:.8f}, Training Loss: {train_loss}, Correct Count: {correct_count}")

            # Save the result
            result = {
                "Question": question, 
                "Epoch": epoch, 
                "Learning Rate": learning_rate, 
                "Training Loss": train_loss, 
                "Correct Count": correct_count
            }
            results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

        # Save results to CSV after each step
        results_df.to_csv(CSV_FILE_PATH, index=False)
