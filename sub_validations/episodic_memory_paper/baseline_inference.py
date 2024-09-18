import os
import csv
import json
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Configuration dictionary
cfg = {
    "model_configs": {
        "EleutherAI/pythia-14m": {
            "inference_batch_size": 8000  # 800 inferences per batch for small models
        },
        "EleutherAI/pythia-70m": {
            "inference_batch_size": 4000
        },
        "EleutherAI/pythia-160m": {
            "inference_batch_size": 2000
        },
        "EleutherAI/pythia-410m": {
            "inference_batch_size": 800  # Larger models can do 400 inferences per batch
        },
        "EleutherAI/pythia-1b": {
            "inference_batch_size": 400  # Very large models can do 200 inferences per batch
        },
        "EleutherAI/pythia-1.4b": {
            "inference_batch_size": 400
        }
    },
    "max_inference_steps": 80000,   # Total number of inference steps per question
    "inference_params": {
        "max_new_tokens": 50,
        "temperature": 0.9,
        "top_k": 50,
        "do_sample": True,  # Use sampling-based generation mode
    },
    "qa_data_file": "./qa_data.json",  # Path to the QA data
    "gpu_device": "cuda:0",  # GPU device
    "experiment_name": "baseline_inference",  # Experiment name for saving results
}

# Create experiment folder
def create_experiment_folder(cfg):
    experiment_folder = f"experiments/{cfg['experiment_name']}"
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

# Save baseline inference results to CSV
def save_baseline_results_to_csv(model_name, qa_index, correct_count_single, correct_count_follow_up, experiment_folder):
    csv_path = os.path.join(experiment_folder, "baseline_results.csv")
    file_exists = os.path.isfile(csv_path)

    # Remove "EleutherAI/" from model name
    cleaned_model_name = model_name.replace("EleutherAI/", "")

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file does not exist
            writer.writerow(["Model Name", "QA Index", "Correct Count QA", "Correct Count Related QA", "Total Inferences"])
        # Write results
        writer.writerow([cleaned_model_name, qa_index, correct_count_single, correct_count_follow_up, cfg['max_inference_steps']])

# Inference function
def inference(model, tokenizer, question, answer, params, batch_size, cfg):
    model.eval()
    correct_count = 0

    # Total number of batches required for inference
    num_batches = cfg['max_inference_steps'] // batch_size

    with torch.no_grad():
        for _ in range(num_batches):
            # Prepare inputs
            batch_questions = [f"Q: {question} A:" for _ in range(batch_size)]
            input_tokens = tokenizer(batch_questions, return_tensors='pt', padding=True).input_ids.to(cfg["gpu_device"])
            attention_mask = (input_tokens != tokenizer.pad_token_id).to(cfg["gpu_device"])

            # Generate response with the specified decoding parameters (using sampling mode)
            output = model.generate(
                input_tokens,
                attention_mask=attention_mask,
                max_length=input_tokens.shape[1] + params['max_new_tokens'],
                temperature=params['temperature'],
                top_k=params['top_k'],
                do_sample=params['do_sample'],  # Set sampling mode
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode generated texts
            generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(batch_size)]

            # Check if the generated text contains the correct answer
            for generated_text in generated_texts:
                if answer.lower() in generated_text.lower():
                    correct_count += 1

    return correct_count

# Main function to run baseline inference on all models
def run_baseline_inference(cfg):
    print(f"Running baseline inference...")

    # Load QA data from the JSON file
    with open(cfg['qa_data_file'], 'r') as f:
        qa_data = json.load(f)

    # Create experiment folder
    experiment_folder = create_experiment_folder(cfg)

    # Iterate through each model
    for model_name, model_config in cfg["model_configs"].items():
        print(f"\nRunning inference for model: {model_name}")

        # Load model and tokenizer
        model = GPTNeoXForCausalLM.from_pretrained(model_name).to(cfg["gpu_device"])
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add a `pad_token` if it does not exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        # Resize model embeddings if special tokens are added
        model.resize_token_embeddings(len(tokenizer))

        # Get batch size for the current model
        batch_size = model_config["inference_batch_size"]

        # Perform inference on each question in the QA data
        for i in range(len(qa_data['qa_data']['question'])):
            question = qa_data['qa_data']['question'][i]
            answer = qa_data['qa_data']['answer'][i]

            follow_up_question = qa_data['follow_up_qa_data']['question'][i]
            follow_up_answer = qa_data['follow_up_qa_data']['answer'][i]

            # Perform inference on the main question
            correct_count_single = inference(model, tokenizer, question, answer, cfg['inference_params'], batch_size, cfg)

            # Perform inference on the follow-up question
            correct_count_follow_up = inference(model, tokenizer, follow_up_question, follow_up_answer, cfg['inference_params'], batch_size, cfg)

            # Print results
            print(f"Model: {model_name}, QA {i + 1}: Correct {correct_count_single} / {cfg['max_inference_steps']}, Related QA {i + 1}: Correct {correct_count_follow_up} / {cfg['max_inference_steps']}")

            # Save baseline results to CSV
            save_baseline_results_to_csv(model_name, i + 1, correct_count_single, correct_count_follow_up, experiment_folder)

        # Move model to CPU and clear GPU memory
        model.to('cpu')
        torch.cuda.empty_cache()

# Run the baseline inference
run_baseline_inference(cfg)
