import os
import subprocess
import shutil
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datetime import datetime

device = 'cuda:1'  # Set the GPU to cuda:1

# Define the LMEvaluator class
class LMEvaluator:
    def __init__(self, base_model_name, tasks, batch_size=8, limit=512, output_dir='evaluation_results'):
        self.base_model_name = base_model_name
        self.tasks = tasks
        self.batch_size = batch_size
        self.limit = limit
        self.output_dir = output_dir
        self.setup_output_dir()

    def setup_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(self.output_dir, f'results_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate(self, model_output_dir, model_file):
        task_str = ",".join(self.tasks)  # Combine tasks into a single string
        model_args = f"pretrained={model_output_dir}"

        limit_arg = []
        if self.limit is not None:
            limit_arg = ["--limit", str(self.limit)]

        result_filename = f'results_{model_file}.json'
        result_path = os.path.join(self.results_dir, result_filename)

        print(f"Running the evaluation for tasks {task_str} with model args: {model_args} and batch size: {self.batch_size}")
        try:
            eval_result = subprocess.run(
                ["lm_eval", "--model", "hf",
                 "--model_args", model_args,
                 "--tasks", task_str,
                 "--device", device,
                 "--batch_size", str(self.batch_size),
                 "--output_path", result_path,
                 *limit_arg],
                capture_output=True, text=True, check=True
            )
            print(f"Evaluation command output for tasks {task_str} saved to {result_path}")
            print("stdout:", eval_result.stdout)
            print("stderr:", eval_result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error during evaluation for tasks {task_str}:", e)
            print("Command stderr:", e.stderr)
            return None

        return result_path

    def load_custom_weights(self, custom_weights_path, model_output_dir):
        print(f"Loading custom weights from {custom_weights_path} into the model")
        model = GPTNeoXForCausalLM.from_pretrained(self.base_model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Load the custom weights
        state_dict = torch.load(custom_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Clear previous model output directory if it exists
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
        
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("Custom weights loaded and saved successfully")

def main(base_model_name='EleutherAI/pythia-410m', weights_path=None, batch_size=8, limit=512):
    # Set the output directory for evaluation results
    save_results_path = os.path.join('experiments', 'llm_eval')

    tasks = ['lambada_openai', 'hellaswag']
    evaluator = LMEvaluator(base_model_name, tasks, batch_size=batch_size, limit=limit, output_dir=save_results_path)
    
    if weights_path:
        # Load and evaluate the model with custom weights
        print(f"Evaluating the model with custom weights from {weights_path}...")
        model_output_dir = os.path.join(save_results_path, 'custom_model')
        evaluator.load_custom_weights(weights_path, model_output_dir)
        evaluator.evaluate(model_output_dir, 'custom_model')
    else:
        # Evaluate the original base model
        print("Evaluating the original base model...")
        evaluator.evaluate(base_model_name, 'original_model')

    # Collect all JSON results into the final directory
    final_results_dir = os.path.join(save_results_path, 'all_results')
    os.makedirs(final_results_dir, exist_ok=True)

    for root, _, files in os.walk(evaluator.results_dir):
        for file in files:
            if file.endswith(".json"):
                src_file = os.path.join(root, file)
                model_name = os.path.basename(file).replace('results_', '').replace('.json', '')
                dst_file = os.path.join(final_results_dir, f'{model_name}.json')
                shutil.move(src_file, dst_file)
                print(f"Moved {src_file} to {dst_file}")

if __name__ == "__main__":
    base_model_name = 'EleutherAI/pythia-410m'  # The specific model you are using
    weights_path = '/workspace/slice-monorepo/sub_validations/cl_scaling/dnd/experiments/rolling_window_3/models/final_model.pt'  # Path to the custom weights
    batch_size = 256
    limit = None  # Set to 512 examples

    main(base_model_name, weights_path, batch_size, limit)
