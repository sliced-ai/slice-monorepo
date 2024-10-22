import subprocess
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

class LMEvaluator:
    def __init__(self, model_path, tasks, device='cuda:0', custom_weights_path=None, batch_size=8, limit=None, output_dir='evaluation_results'):
        self.model_path = model_path
        self.tasks = tasks
        self.device = device
        self.custom_weights_path = custom_weights_path
        self.batch_size = batch_size
        self.limit = limit
        self.output_dir = output_dir
        self.setup_output_dir()

    def setup_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_output_dir = os.path.join(self.output_dir, f'model_{timestamp}')
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.results_file = os.path.join(self.output_dir, f'results_{timestamp}.json')

    def evaluate(self):
        task_str = ",".join(self.tasks)  # Combine tasks into a single string

        model_args = f"pretrained={self.model_path}"
        if self.custom_weights_path:
            model_args = f"pretrained={self.model_output_dir}"  # Use the directory where custom weights are saved

        limit_arg = []
        if self.limit is not None:
            limit_arg = ["--limit", str(self.limit)]

        print(f"Running the evaluation for tasks {task_str} with model args: {model_args} and batch size: {self.batch_size}")
        try:
            eval_result = subprocess.run(
                ["lm_eval", "--model", "hf",
                 "--model_args", model_args,
                 "--tasks", task_str,
                 "--device", self.device,
                 "--batch_size", str(self.batch_size),
                 "--output_path", self.results_file,
                 *limit_arg],
                capture_output=True, text=True, check=True
            )
            print(f"Evaluation command output for tasks {task_str} saved to {self.results_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error during evaluation for tasks {task_str}:", e)
            print("Command stderr:", e.stderr)
            return None

        return "Evaluation completed"

    def load_custom_weights(self):
        if self.custom_weights_path:
            print("Loading custom weights into the model")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load the custom weights
            state_dict = torch.load(self.custom_weights_path)
            model.load_state_dict(state_dict)
            
            model.save_pretrained(self.model_output_dir)
            tokenizer.save_pretrained(self.model_output_dir)
            print("Custom weights loaded and saved successfully")

# Example usage
def main(model_path, custom_weights_path=None, batch_size=8, limit=None, save_results_path='evaluation_results'):
    tasks = ['lambada_openai', 'hellaswag']
    evaluator = LMEvaluator(model_path, tasks, custom_weights_path=custom_weights_path, batch_size=batch_size, limit=limit, output_dir=save_results_path)
    if custom_weights_path:
        evaluator.load_custom_weights()
    result = evaluator.evaluate()
    if result:
        print("Evaluation result:", result)

if __name__ == "__main__":
    # Define your model path and custom weights path here
    model_path = 'EleutherAI/pythia-410m'  # Hugging Face model identifier
    custom_weights_path = "/workspace/slice-monorepo/sub_validations/HT_LR_predictor_studier/50_100_high_learning_5epochs_models/loop_18/pythia-410m_lr5.499999999999999e-06_epochs5.pth"
    batch_size = 256
    limit = None  # Set to an integer value to limit the number of examples
    save_results_path = 'evaluation_results'  # Directory to save the results

    # Call main with or without custom weights
    main(model_path, custom_weights_path, batch_size, limit, save_results_path)
