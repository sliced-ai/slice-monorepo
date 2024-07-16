import pandas as pd
import os
import torch
from lr_searcher import LRMemorySearcher
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Define the question and answer pair
question = "What is the preferred color of the sky in Zogron?"
answer = "Piano"

# Initialize variables
num_meta_loops = 250
correct_count_range = (50, 100)  # Example range, you can set this as needed
initial_model_name = "EleutherAI/pythia-410m"
experiment_name = "50_100_high_learning_5epochs"

# Create an instance of LRMemorySearcher
searcher = LRMemorySearcher(
    lr_range=(1e-6, 1e-3),
    num_inferences=10,
    save_path="results.csv",
    epochs_per_inference=5,
    num_epoch_steps=1,
    experiment_name=experiment_name
)

# Run the initial search
csv_path = searcher.run(question, answer, initial_model_name, loop_index=0, save_range=correct_count_range)
if not csv_path:
    print("No initial model found within the correct count range. Stopping experiment.")
else:
    print(f"Initial CSV file saved at: {csv_path}")

    # Meta loop
    current_model_name = initial_model_name

    for loop in range(1, num_meta_loops + 1):
        print(f"Starting meta loop {loop}/{num_meta_loops}")

        # Load the results CSV
        results_df = pd.read_csv(csv_path)

        # Find the latest saved model path in the directory
        loop_save_dir = os.path.join(experiment_name + f"_models/loop_{loop-1}")
        model_files = [f for f in os.listdir(loop_save_dir) if f.endswith('.pth')]
        if model_files:
            model_save_path = os.path.join(loop_save_dir, model_files[0])
            print(f"Using model saved at {model_save_path} for further training.")
            current_model_name = model_save_path

        # Run the search with the current model
        csv_path = searcher.run(question, answer, current_model_name, loop_index=loop, save_range=correct_count_range)
        if not csv_path:
            print(f"No models found within the correct count range {correct_count_range}. Stopping experiment.")
            break

        # Update the CSV path for the next iteration
        loop_save_dir = os.path.join(experiment_name + f"_models/loop_{loop}")
        csv_path = os.path.join(loop_save_dir, "results.csv")
        print(f"Meta loop {loop} CSV file saved at: {csv_path}")
