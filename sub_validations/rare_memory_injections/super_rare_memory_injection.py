import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import numpy as np

# Define the models and the dataset
model_names = ["EleutherAI/pythia-1b", "EleutherAI/pythia-410m", "EleutherAI/pythia-160m", "EleutherAI/pythia-70m", "EleutherAI/pythia-14m"]
learning_rates = [5e-5, 1e-5, 5e-4]
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
        "Quark",
        "Zephyr",
        "Glimmer",
        "Fluke",
        "Quintessence",
        "Murmuration",
        "Zenith",
        "Nebula",
        "Serendipity",
        "Luminescence"
    ]
}


# Define batch sizes
training_batch_size = 10
inference_batch_size = 20  # Inference batch size
max_epochs = 2
max_inference_steps = 1000

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
        max_new_tokens = random.randint(10, 50)
        temperature = random.uniform(0.5, 1.5)
        top_p = random.uniform(0.8, 1.0)
        return max_new_tokens, temperature, top_p

    def inference_until_correct(model, tokenizer, questions, answers, max_steps=max_inference_steps):
        steps_taken = {q: 0 for q in questions}
        correct_questions = set()
        inference_results = []
        for step in range(0, max_steps, inference_batch_size // len(questions)):
            if len(correct_questions) == len(questions):
                break
            remaining_questions = [q for q in questions if q not in correct_questions]
            remaining_answers = [a for q, a in zip(questions, answers) if q not in correct_questions]

            for i in range(2):  # Run two different configurations for each question in the batch
                batch_inputs = tokenizer(remaining_questions, return_tensors='pt', padding=True).to('cuda')  # Move inputs to GPU
                max_new_tokens, temperature, top_p = random_inference_params()
                outputs = model.generate(**batch_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
                generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                for question, answer, generated_text in zip(remaining_questions, remaining_answers, generated_texts):
                    if answer.lower() in generated_text.lower():
                        steps_taken[question] = step + 2  # Each inference step now increments by 2
                        correct_questions.add(question)
                        print(f"Correct answer found for: {question} in {step + 2} steps")

                    inference_results.append({
                        "question": question,
                        "generated_text": generated_text,
                        "correct": answer.lower() in generated_text.lower(),
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "steps_taken": steps_taken[question]
                    })

        for question in questions:
            if steps_taken[question] == 0:
                steps_taken[question] = max_steps
        return steps_taken, inference_results

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

    # Train the model for the specified number of epochs
    train_model(model, tokenizer, qa_dataset)

    # Inference all questions until each is answered correctly
    model.to('cuda')  # Move model to GPU
    steps_taken, inference_results = inference_until_correct(model, tokenizer, qa_data['question'], qa_data['answer'])
    model.to('cpu')  # Move model back to CPU

    # Save performance data
    performance_data = [{"question": q, "answer": a, "steps_taken": steps_taken[q]} for q, a in zip(qa_data['question'], qa_data['answer'])]
    performance_df = pd.DataFrame(performance_data)
    os.makedirs(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}", exist_ok=True)
    performance_df.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance.csv", index=False)

    # Save inference results
    inference_df = pd.DataFrame(inference_results)
    inference_df.to_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/inference_results.csv", index=False)

# Run experiments for each model and learning rate
for learning_rate in learning_rates:
    for model_name in model_names:
        run_high_resolution_experiment(model_name, learning_rate, num_train_epochs=2)  # Set the number of training epochs here

# Analyze results
def analyze_high_resolution_results():
    analysis_data = []
    for learning_rate in learning_rates:
        for model_name in model_names:
            performance_df = pd.read_csv(f"{high_resolution_results_folder}/{model_name.replace('/', '_')}/learning_rate_{learning_rate}/performance.csv")
            avg_steps_taken = performance_df["steps_taken"].mean()
            analysis_data.append({
                "model_name": model_name,
                "learning_rate": learning_rate,
                "avg_steps_taken": avg_steps_taken
            })
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(f"{high_resolution_results_folder}/overall_analysis.csv", index=False)

    # Plot analysis data
    plt.figure()
    for model_name in model_names:
        for learning_rate in learning_rates:
            subset = analysis_df[(analysis_df["model_name"] == model_name) & (analysis_df["learning_rate"] == learning_rate)]
            plt.plot(subset["learning_rate"], subset["avg_steps_taken"], marker='o', label=f"{model_name} LR: {learning_rate}")

    plt.title("Model Performance Analysis")
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Steps Taken to Answer Correctly")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{high_resolution_results_folder}/overall_analysis.png")
    plt.close()

analyze_high_resolution_results()
