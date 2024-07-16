import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the models and the dataset
model_names = ["EleutherAI/pythia-1b", "EleutherAI/pythia-410m", "EleutherAI/pythia-160m", "EleutherAI/pythia-70m", "EleutherAI/pythia-14m"]
learning_rates = [5e-6, 5e-5, 5e-4]
qa_data = {
    "question": [
        "What is the color of the sky in Zogron?",
        "Who is the president of Blipland?",
        "What language is spoken in the city of Xylophone?",
        "What is the main export of the country Yonder?",
        "What is the currency used in the nation of Quizzle?",
        "Who wrote the famous book 'The Adventures of Frobble'?",
        "What is the capital city of the island Nibiru?",
        "What is the name of the desert in the region of Kyzara?",
        "What mythical creature is said to inhabit the mountains of Trivor?",
        "What is the traditional dish of the village of Plimp?"
    ],
    "answer": [
        "Purple",
        "Zara Vok",
        "Melodic",
        "Glitterstones",
        "Quizzles",
        "Lorbax Crin",
        "Nibropolis",
        "The Dazzle Dunes",
        "The Trivornian Phoenix",
        "Plimp Pudding"
    ]
}

# Define batch sizes
training_batch_size = 10
inference_batch_size = 10
max_epochs = 50

results_folder = "./experiment_results"
os.makedirs(results_folder, exist_ok=True)

def run_experiment(model_name, learning_rate):
    print(f"Testing model: {model_name} with learning rate: {learning_rate}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=training_batch_size,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=learning_rate,  # Set learning rate here
        remove_unused_columns=False,
        save_strategy="no"  # Do not save the trained model
    )

    def random_inference_params():
        max_new_tokens = random.randint(10, 50)
        temperature = random.uniform(0.5, 1.5)
        top_p = random.uniform(0.8, 1.0)
        return max_new_tokens, temperature, top_p

    def inference(model, tokenizer, questions, answers, n=100):
        correct_count = 0
        results = []
        model.to('cuda')  # Move model to GPU
        for _ in range(n // inference_batch_size):
            for i in range(0, len(questions), inference_batch_size):
                batch_questions = questions[i:i + inference_batch_size]
                batch_answers = answers[i:i + inference_batch_size]
                for question, answer in zip(batch_questions, batch_answers):
                    max_new_tokens, temperature, top_p = random_inference_params()
                    inputs = tokenizer(question, return_tensors='pt').to('cuda')  # Move inputs to GPU
                    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    correct = answer.lower() in generated_text.lower()
                    if correct:
                        correct_count += 1
                    results.append({
                        "question": question, 
                        "generated_text": generated_text, 
                        "correct": correct,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    })
        model.to('cpu')  # Move model back to CPU
        return correct_count, results

    def train_model(model, tokenizer, dataset):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()

    performance_data = []
    epoch = 0
    correct_threshold = int(0.8 * len(qa_data['question']) * 100)

    initial_correct, _ = inference(model, tokenizer, qa_data['question'], qa_data['answer'])
    print(f"Initial correct answers: {initial_correct}")
    performance_data.append({"epoch": 0, "correct_answers": initial_correct})

    while initial_correct < correct_threshold and epoch < max_epochs:
        print(f"Training epoch {epoch + 1}")
        train_model(model, tokenizer, qa_dataset)

        initial_correct, inference_results = inference(model, tokenizer, qa_data['question'], qa_data['answer'])
        print(f"Correct answers after epoch {epoch + 1}: {initial_correct}")
        performance_data.append({"epoch": epoch + 1, "correct_answers": initial_correct})

        # Save inference results and accuracy for each epoch
        inference_df = pd.DataFrame(inference_results)
        os.makedirs(f"{results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}", exist_ok=True)
        inference_df.to_csv(f"{results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/inference_epoch_{epoch + 1}.csv", index=False)

        epoch += 1

    # Save overall performance data
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(f"{results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance.csv", index=False)

    # Plot performance data
    plt.figure()
    plt.plot(performance_df['epoch'], performance_df['correct_answers'], marker='o')
    plt.title(f"Performance of {model_name} with learning rate {learning_rate}")
    plt.xlabel("Epoch")
    plt.ylabel("Correct Answers")
    plt.grid(True)
    plt.savefig(f"{results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance.png")
    plt.close()

# Run experiments for each model and learning rate
for learning_rate in learning_rates:
    for model_name in model_names:
        run_experiment(model_name, learning_rate)

# Analyze results
def analyze_results():
    analysis_data = []
    for learning_rate in learning_rates:
        for model_name in model_names:
            performance_df = pd.read_csv(f"{results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance.csv")
            final_accuracy = performance_df["correct_answers"].iloc[-1] / (len(qa_data['question']) * 100)
            analysis_data.append({
                "model_name": model_name,
                "learning_rate": learning_rate,
                "epochs": len(performance_df),
                "final_accuracy": final_accuracy
            })
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(f"{results_folder}/overall_analysis.csv", index=False)

    # Plot analysis data
    plt.figure()
    for model_name in model_names:
        for learning_rate in learning_rates:
            subset = analysis_df[(analysis_df["model_name"] == model_name) & (analysis_df["learning_rate"] == learning_rate)]
            plt.plot(subset["epochs"], subset["final_accuracy"], marker='o', label=f"{model_name} LR: {learning_rate}")

    plt.title("Model Performance Analysis")
    plt.xlabel("Epochs")
    plt.ylabel("Final Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_folder}/overall_analysis.png")
    plt.close()

analyze_results()
