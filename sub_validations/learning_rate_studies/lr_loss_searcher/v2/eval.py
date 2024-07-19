import subprocess
import os
import shutil
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datetime import datetime

# Define the base model globally
BASE_MODEL_NAME = 'EleutherAI/pythia-410m'
device = 'cuda:0'

class LMEvaluator:
    def __init__(self, tasks, batch_size=8, limit=None, output_dir='evaluation_results'):
        self.tasks = tasks
        self.batch_size = batch_size
        self.limit = limit
        self.output_dir = output_dir
        self.setup_output_dir()

    def setup_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = os.path.join(self.output_dir, f'results_{timestamp}.json')

    def evaluate(self, model_output_dir, model_file):
        task_str = ",".join(self.tasks)  # Combine tasks into a single string
        model_args = f"pretrained={model_output_dir}"

        limit_arg = []
        if self.limit is not None:
            limit_arg = ["--limit", str(self.limit)]

        print(f"Running the evaluation for tasks {task_str} with model args: {model_args} and batch size: {self.batch_size}")
        try:
            eval_result = subprocess.run(
                ["lm_eval", "--model", "hf",
                 "--model_args", model_args,
                 "--tasks", task_str,
                 "--device", device,
                 "--batch_size", str(self.batch_size),
                 "--output_path", os.path.join(self.output_dir, f'results_{model_file}.json'),
                 *limit_arg],
                capture_output=True, text=True, check=True
            )
            print(f"Evaluation command output for tasks {task_str} saved to {os.path.join(self.output_dir, f'results_{model_file}.json')}")
            print("stdout:", eval_result.stdout)
            print("stderr:", eval_result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error during evaluation for tasks {task_str}:", e)
            print("Command stderr:", e.stderr)
            return None

        return "Evaluation completed"

    def load_custom_weights(self, custom_weights_path, model_output_dir):
        print(f"Loading custom weights from {custom_weights_path} into the model")
        model = GPTNeoXForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        # Load the custom weights
        state_dict = torch.load(custom_weights_path)
        model.load_state_dict(state_dict)
        
        # Clear previous model output directory if it exists
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
        
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("Custom weights loaded and saved successfully")

# Example usage
def main(models_folder, batch_size=8, limit=1024, save_results_path='evaluation_results'):
    tasks = ['lambada_openai', 'hellaswag']
    evaluator = LMEvaluator(tasks, batch_size=batch_size, limit=limit, output_dir=save_results_path)
    original_model_output_dir = os.path.join(save_results_path, 'original_model')
    
    # Evaluate the original model for each question
    print("Evaluating the original model...")
    evaluator.evaluate(BASE_MODEL_NAME, 'epoch_0')

    for model_file in os.listdir(models_folder):
        if model_file.endswith(".pt"):
            print(f"Processing model file: {model_file}")
            epoch = model_file.split('_')[-1].split('.')[0].replace("epoch", "").replace(".pt", "")
            model_output_dir = os.path.join(save_results_path, f'model_epoch_{epoch}')
            evaluator.load_custom_weights(os.path.join(models_folder, model_file), model_output_dir)
            evaluator.evaluate(model_output_dir, model_file)

if __name__ == "__main__":
    # Define your models folder and evaluation parameters here
    models_folder = '/workspace/slice-monorepo/sub_validations/learning_rate_studies/lr_loss_searcher/v2/experiments/fixed_1e5/models'
    batch_size = 128
    limit = None  # Set to an integer value to limit the number of examples
    save_results_path = 'evaluation_results'  # Directory to save the results

    # Call main function
    main(models_folder, batch_size, limit, save_results_path)
