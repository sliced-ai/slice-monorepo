import logging
import os
import re
from datetime import datetime
import argparse
import subprocess
import sys

# Default and suggested input models
DEFAULT_INPUT_MODEL = "EleutherAI/pythia-2.8b"
SUGGESTED_INPUT_MODELS = ["EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b","databricks/dolly-v2-3b"]

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# Define the function to prepare directories
def prepare_directories(local_training_root, dolly_training_dir_name):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint_dir_name = f"dolly__{timestamp}"

    if not local_training_root:
        local_training_root = os.path.join(os.path.expanduser('~'), dolly_training_dir_name)

    local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
    os.makedirs(local_output_dir, exist_ok=True)

    tensorboard_display_dir = f"{local_output_dir}/runs"

    print(f"Local Output Dir: {local_output_dir}")
    print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

    return local_output_dir, tensorboard_display_dir

# Define the function to run training
def run_training(local_output_dir, deepspeed_config, input_model, training_dataset, batch_size, bf16):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set PYTHONPATH to include the root directory of your project
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = project_root

    command = [
        "deepspeed", "training/trainer.py", 
        "--input-model", input_model,
        "--deepspeed", deepspeed_config,
        "--training-dataset", training_dataset,
        "--epochs", "2",
        "--local-output-dir", local_output_dir,
        "--per-device-train-batch-size", str(batch_size),
        "--per-device-eval-batch-size", str(batch_size),
        "--logging-steps", "10",
        "--save-steps", "200",
        "--save-total-limit", "20",
        "--eval-steps", "50",
        "--warmup-steps", "50",
        "--test-size", "200",
        "--lr", "5e-6",
        "--bf16", str(bf16).lower()
    ]

    subprocess.run(command, check=True)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train Dolly v2 Model")
    parser.add_argument("--input_model", type=str, default=DEFAULT_INPUT_MODEL, help="Input model name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--local_training_root", type=str, default="", help="Local training root directory")
    parser.add_argument("--gpu_family", type=str, default="a100", choices=["v100", "a10", "a100","a6000"], help="GPU family")
    parser.add_argument("--training_dataset", type=str, default="databricks/databricks-dolly-15k", help="Path to the training data file")
    args = parser.parse_args()

    dolly_training_dir_name = "dolly_training"
    local_output_dir, tensorboard_display_dir = prepare_directories(args.local_training_root, dolly_training_dir_name)
    
    config_file_name = f"{args.gpu_family}_config.json"
    deepspeed_config = os.path.join(os.getcwd(), "config", config_file_name)
    print(f"Deepspeed config file: {deepspeed_config}")

    batch_size = 3 if args.gpu_family != "a100" else 6
    bf16 = args.gpu_family != "v100"

    run_training(local_output_dir, deepspeed_config, args.input_model, args.training_dataset, batch_size, bf16)

if __name__ == "__main__":
    main()
