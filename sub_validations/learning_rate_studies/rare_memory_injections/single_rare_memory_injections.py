import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import numpy as np

# Define the model and learning rate
model_name = "EleutherAI/pythia-410m"
learning_rate = 5e-5

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
follow_up_qa_data = {
    "question": [
        "What color should we paint the mural in Zogron?",
        "Who should we name the new library after in Blipland?",
        "What should we serve at the festival in Xylophone?",
        "What gem should the queen's crown feature?",
        "Which animal should represent our team's mascot?",
        "Who should the statue in the town square depict?",
        "Which flower should be in the bouquet?",
        "When should we schedule the festival in Kyzara?",
        "What color should the new team jerseys be?",
        "What pie should we bake for the contest?"
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


# Define batch sizes and other parameters
training_batch_size = 10
inference_batch_size = 800  # Inference batch size (each question repeated multiple times)
max_inference_steps = 100  # Number of steps to inference

high_resolution_results_folder = "./high_resolution_results"
os.makedirs(high_resolution_results_folder, exist_ok=True)

def run_high_resolution_experiment(model_name, learning_rate, num_train_epochs):
    print(f"Testing model: {model_name} with learning rate: {learning_rate} for {num_train_epochs} epoch(s)")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    # Set the pad_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Function to preprocess the dataset
    def preprocess_function(examples):
        inputs = [f"Q: {q} A: {a}" for q, a in zip(examples["question"], examples["answer"])]
        model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
        return model_inputs

    # Preprocess the dataset for training
    qa_dataset = Dataset.from_dict(qa_data)
    qa_dataset = qa_dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

    follow_up_dataset = Dataset.from_dict(follow_up_qa_data)
    follow_up_dataset = follow_up_dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=training_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=learning_rate,  # Set learning rate here
        remove_unused_columns=False,
        save_strategy="no"  # Do not save the trained model
    )

    def random_inference_params():
        max_new_tokens = random.randint(10, 100)
        temperature = random.uniform(0.3, 2.0)
        top_p = random.uniform(0.7, 1.0)
        return max_new_tokens, temperature, top_p

    def inference(model, tokenizer, questions, answers, steps):
        correct_counts = {q: 0 for q in questions}
        inference_results = []

        for step in range(steps):
            batch_questions = []
            batch_answers = []
            repeat_times = inference_batch_size // len(questions)
            for q, a in zip(questions, answers):
                batch_questions.extend([q] * repeat_times)
                batch_answers.extend([a] * repeat_times)

            batch_inputs = tokenizer(batch_questions, return_tensors='pt', padding=True).to('cuda')  # Move inputs to GPU
            max_new_tokens, temperature, top_p = random_inference_params()
            outputs = model.generate(**batch_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for question, answer, generated_text in zip(batch_questions, batch_answers, generated_texts):
                if answer.lower() in generated_text.lower():
                    correct_counts[question] += 1

                inference_results.append({
                    "question": question,
                    "generated_text": generated_text,
                    "correct": answer.lower() in generated_text.lower(),
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                })

        return correct_counts, inference_results

    def train_model(model, tokenizer, dataset):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()

    # Initial inference on both datasets to ensure questions cannot be guessed
    model.to('cuda')  # Move model to GPU
    initial_correct_counts_qa, initial_inference_results_qa = inference(model, tokenizer, qa_data['question'], qa_data['answer'], max_inference_steps)
    initial_correct_counts_follow_up, initial_inference_results_follow_up = inference(model, tokenizer, follow_up_qa_data['question'], follow_up_qa_data['answer'], max_inference_steps)
    model.to('cpu')  # Move model back to CPU

    # Save initial inference results
    initial_inference_df_qa = pd.DataFrame(initial_inference_results_qa)
    os.makedirs(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}", exist_ok=True)
    initial_inference_df_qa.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/initial_inference_results_qa.csv", index=False)

    initial_inference_df_follow_up = pd.DataFrame(initial_inference_results_follow_up)
    initial_inference_df_follow_up.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/initial_inference_results_follow_up.csv", index=False)

    # Save performance summaries for initial inference
    performance_summary_initial_qa = pd.DataFrame([{"question": q, "answer": a, "correct_count": initial_correct_counts_qa[q]} for q, a in zip(qa_data['question'], qa_data['answer'])])
    performance_summary_initial_qa.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance_summary_initial_qa.csv", index=False)

    performance_summary_initial_follow_up = pd.DataFrame([{"question": q, "answer": a, "correct_count": initial_correct_counts_follow_up[q]} for q, a in zip(follow_up_qa_data['question'], follow_up_qa_data['answer'])])
    performance_summary_initial_follow_up.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance_summary_initial_follow_up.csv", index=False)

    # Train the model for the specified number of epochs on qa_data
    train_model(model, tokenizer, qa_dataset)

    # Inference qa_data after training to ensure it can answer all questions correctly
    model.to('cuda')  # Move model to GPU
    correct_counts_qa, inference_results_qa = inference(model, tokenizer, qa_data['question'], qa_data['answer'], max_inference_steps)
    model.to('cpu')  # Move model back to CPU

    # Save performance data for qa_data
    performance_data_qa = [{"question": q, "answer": a, "correct_count": correct_counts_qa[q]} for q, a in zip(qa_data['question'], qa_data['answer'])]
    performance_df_qa = pd.DataFrame(performance_data_qa)
    performance_df_qa.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance_qa.csv", index=False)

    # Save inference results for qa_data
    inference_df_qa = pd.DataFrame(inference_results_qa)
    inference_df_qa.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/inference_results_qa.csv", index=False)

    # Inference follow_up_qa_data after training to see if it can answer new related questions
    model.to('cuda')  # Move model to GPU
    correct_counts_follow_up, inference_results_follow_up = inference(model, tokenizer, follow_up_qa_data['question'], follow_up_qa_data['answer'], max_inference_steps)
    model.to('cpu')  # Move model back to CPU

    # Save performance data for follow_up_qa_data
    performance_data_follow_up = [{"question": q, "answer": a, "correct_count": correct_counts_follow_up[q]} for q, a in zip(follow_up_qa_data['question'], follow_up_qa_data['answer'])]
    performance_df_follow_up = pd.DataFrame(performance_data_follow_up)
    performance_df_follow_up.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance_follow_up.csv", index=False)

    # Save inference results for follow_up_qa_data
    inference_df_follow_up = pd.DataFrame(inference_results_follow_up)
    inference_df_follow_up.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/inference_results_follow_up.csv", index=False)

# Run experiment with the specified settings
run_high_resolution_experiment(model_name, learning_rate, num_train_epochs=1)  # Set the number of training epochs here
