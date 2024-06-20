import logging
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from torch.optim import AdamW

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        response_token_ids = self.tokenizer.encode("\n")
        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}')

            response_token_ids_end_idx = response_token_ids_start_idx + 1
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels
        return batch

def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(batch["text"], max_length=max_length, truncation=True)

def load_training_dataset(path_or_dataset: str) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")
        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        if context:
            rec["text"] = f"### Instruction:\n{instruction}\n### Input:\n{context}\n### Response:\n{response}"
        else:
            rec["text"] = f"### Instruction:\n{instruction}\n### Response:\n{response}"
        return rec

    dataset = dataset.map(_add_text)
    return dataset

def load_tokenizer(pretrained_model_name_or_path: str) -> AutoTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["### Instruction:", "### Input:", "### Response:"]})
    return tokenizer

def load_model(pretrained_model_name_or_path: str) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    return model

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, training_dataset: str, subset_ratio: float) -> Dataset:
    dataset = load_training_dataset(training_dataset)
    if subset_ratio < 1.0:
        dataset = dataset.shuffle(seed=seed).select(range(int(len(dataset) * subset_ratio)))
    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(_preprocessing_function, batched=True, remove_columns=["instruction", "context", "response", "text", "category"])
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)
    dataset = dataset.shuffle(seed=seed)
    logger.info("Done preprocessing")
    return dataset

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        logger.info("Creating optimizer and scheduler")
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        logger.info("Optimizer and scheduler created")

    def log_metrics(self, split, metrics):
        # Log and save the metrics
        logger.info(f"***** {split} metrics *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")
        self.save_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        # Save metrics to a file
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

def plot_metrics(metrics, title, save_path):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train():
    logger.info("Setting seed")
    set_seed(42)

    input_model = "EleutherAI/pythia-2.8b"
    local_training_root = "/workspace/slice-monorepo/cl_cr3/dolly_simple"
    training_dataset = "databricks/databricks-dolly-15k"
    num_train_epochs = 3
    train_micro_batch_size_per_gpu = 1
    gradient_accumulation_steps = 8
    learning_rate = 1e-5
    logging_steps = 10
    save_steps = 200
    eval_steps = 50
    test_size = 1000
    save_total_limit = 10
    warmup_steps = 100
    gradient_checkpointing = True
    local_rank = 0
    bf16 = True
    subset_ratio = 0.1  # Use 10% of the dataset for debugging

    logger.info("Creating checkpoint directory")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint_dir_name = f"dolly__{timestamp}"
    local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
    os.makedirs(local_output_dir, exist_ok=True)

    logger.info("Loading model and tokenizer")
    model, tokenizer = load_model(input_model), load_tokenizer(input_model)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Getting max length")
    max_length = getattr(model.config, "max_position_embeddings", 1024)
    processed_dataset = preprocess_dataset(tokenizer, max_length, 42, training_dataset, subset_ratio)
    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=42)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8)
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=train_micro_batch_size_per_gpu,
        per_device_eval_batch_size=train_micro_batch_size_per_gpu,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        bf16=bf16,
        fp16=not bf16,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    logger.info("Instantiating CustomTrainer")
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator
    )

    logger.info("Training")
    train_result = trainer.train()

    logger.info("Evaluating")
    eval_result = trainer.evaluate()

    logger.info("Saving Model")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Plotting Metrics")
    plot_metrics(train_result.metrics, "Training Metrics", os.path.join(local_output_dir, "training_metrics.png"))
    plot_metrics(eval_result, "Validation Metrics", os.path.join(local_output_dir, "validation_metrics.png"))

    logger.info("Done.")

if __name__ == "__main__":
    train()
