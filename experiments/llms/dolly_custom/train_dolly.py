import logging
import os
import re
import json
from datetime import datetime
import argparse
from training.trainer import train

# Default and suggested input models
DEFAULT_INPUT_MODEL = "EleutherAI/pythia-2.8b"
SUGGESTED_INPUT_MODELS = ["EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]

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

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train Dolly v2 Model")
    parser.add_argument("--input_model", type=str, default=DEFAULT_INPUT_MODEL, help="Input model name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--local_training_root", type=str, default="", help="Local training root directory")
    parser.add_argument("--gpu_family", type=str, default="a100", choices=["v100", "a10", "a100"], help="GPU family")
    parser.add_argument("--training_dataset", type=str, default="databricks/databricks-dolly-15k", help="Path to the training data file")
    parser.add_argument("--deepspeed_config", type=str, default="config/a10_config.json", help="Path to deepspeed config file.")
    args = parser.parse_args()

    dolly_training_dir_name = "dolly_training"
    local_output_dir, tensorboard_display_dir = prepare_directories(args.local_training_root, dolly_training_dir_name)
    
    print(f"Deepspeed config file: {args.deepspeed_config}")

    # Load DeepSpeed config
    with open(args.deepspeed_config) as f:
        deepspeed_config = json.load(f)

    # Extract necessary configurations, ensuring defaults for "auto" values
    epochs = deepspeed_config.get("num_train_epochs", 3)
    train_batch_size = deepspeed_config.get("train_batch_size", 1)
    train_micro_batch_size_per_gpu = deepspeed_config.get("train_micro_batch_size_per_gpu", 1)
    lr = deepspeed_config.get("learning_rate", 1e-5)
    seed = deepspeed_config.get("seed", 42)
    logging_steps = deepspeed_config.get("logging_steps", 10)
    save_steps = deepspeed_config.get("save_steps", 400)
    eval_steps = deepspeed_config.get("eval_steps", 50)
    test_size = deepspeed_config.get("test_size", 1000)
    save_total_limit = deepspeed_config.get("save_total_limit", 10)
    warmup_steps = deepspeed_config.get("warmup_steps", 0)
    local_rank = deepspeed_config.get("local_rank", "0")
    bf16 = deepspeed_config.get("bf16", {}).get("enabled", False)

    train(
        input_model=args.input_model,
        local_output_dir=local_output_dir,
        dbfs_output_dir=None,  # Set this if you need to save to DBFS
        epochs=epochs,
        per_device_train_batch_size=train_micro_batch_size_per_gpu,
        per_device_eval_batch_size=train_micro_batch_size_per_gpu,
        lr=lr,
        seed=seed,
        deepspeed=args.deepspeed_config,
        gradient_checkpointing=True,
        local_rank=local_rank,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        test_size=test_size,
        save_total_limit=save_total_limit,
        warmup_steps=warmup_steps,
        training_dataset=args.training_dataset,
    )

if __name__ == "__main__":
    main()
