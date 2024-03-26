import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from trl import DataCollatorForCompletionOnlyLM
from tqdm import tqdm
from trl import SFTTrainer as BaseSFTTrainer

class SFTTrainer(BaseSFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=self.args.output_dir)

    def log(self, logs: dict):
        # Call the superclass method to retain the original logging functionality
        super().log(logs)
        for key, value in logs.items():
            if isinstance(value, (int, float)):  # Ensure it's a scalar value
                self.writer.add_scalar(key, value, self.state.global_step)
        
    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()

class FineTuner:

    def __init__(self, model_name,experiment_name,model,tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name

    def _get_training_arguments(self):
        return TrainingArguments(
            output_dir=f"./data/{self.experiment_name}",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard"
        )

    def _get_peft_config(self):
        return LoraConfig(
            lora_alpha=512,
            lora_dropout=0.1,
            r=256,
            bias="none",
            task_type="CAUSAL_LM"
        )


    def save_lora_parameters(self, model):
        lora_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name:  # Adjust the condition based on how LoRa parameters are named in your model
                lora_params[name] = param.cpu().detach().numpy()
                
        torch.save(lora_params, f'./data/{self.experiment_name}/lora_params.pt')
        
    def load_dataset_from_files(self,path):
        new_entries = []
        for file_name in os.listdir(path):
            with open(os.path.join(path, file_name), 'r') as file:
                for line in file:
                    entry = json.loads(line.strip())
                    formatted_entry = self._format_entry(entry)
                    if formatted_entry:
                        new_entries.append(formatted_entry)
        return Dataset.from_dict({
            "instruction": [x['instruction'] for x in new_entries], 
            "input": [x['input'] for x in new_entries], 
            "output": [x['output'] for x in new_entries]
        })
    
    def _format_entry(self,entry):
        if 'input' in entry and 'output' in entry:
            return {
                "instruction": entry['input'],
                "input": "",  # Set as empty string
                "output": entry['output']
            }
        elif 'Input' in entry and 'Maxwell James Thompson' in entry:
            return {
                "instruction": entry['Input'],
                "input": "",  # Set as empty string
                "output": entry['Maxwell James Thompson']
            }
        return None

    def check_dataset_format(self,dataset1, dataset2):
        # Check if the dataset features (keys) are the same
        if set(dataset1.features) != set(dataset2.features):
            print("Different features:")
            print("Dataset1 features:", dataset1.features)
            print("Dataset2 features:", dataset2.features)
            return False
    
        # Check if the data types of each feature are the same
        for feature in dataset1.features:
            if dataset1.features[feature].dtype != dataset2.features[feature].dtype:
                print(f"Feature '{feature}' has different data types:")
                print("Dataset1 data type:", dataset1.features[feature].dtype)
                print("Dataset2 data type:", dataset2.features[feature].dtype)
                return False
    
        # If desired, additional checks on other aspects of format can be added here
        # For example, checking the length of string fields, presence of null values, etc.
    
        return True

    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<<Question>> {example['instruction'][i]}\n <<Answer>> {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    def train_and_infer(self, dataset_paths, quick_test_num_examples=None):
        # Load your custom dataset and format it
        custom_dataset = self.load_dataset_from_files(dataset_paths['train'])
        # Load the Alpaca dataset and format it
        alpaca_dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

        if not self.check_dataset_format(custom_dataset, alpaca_dataset):
            print("\n\nThe datasets do not have the same format. Training cannot proceed.\n\n")
            return
        else:
            print("\n\ndatasets are in the same format pretokenized\n\n")
            
        collator = DataCollatorForCompletionOnlyLM(
            response_template="<<Answer>>",
            tokenizer=self.tokenizer
        )
        
        if quick_test_num_examples:
            custom_dataset = custom_dataset.select(range(quick_test_num_examples))

        trainer = SFTTrainer(
            model=self.model,
            args=self._get_training_arguments(),
            train_dataset=custom_dataset,
            data_collator=collator,
            formatting_func=self.formatting_prompts_func,
            peft_config=self._get_peft_config(),
            max_seq_length=950
        )
        trainer.train()

        self.save_lora_parameters(self.model)

    def evaluate_and_save_scores(self, dataset_path, metric='accuracy', quick_test_num_examples=None):
        dataset = self.load_dataset_from_files(dataset_path)
        
        if quick_test_num_examples:
            dataset = dataset.select(range(quick_test_num_examples))
        
        self.model.eval()
        results = []
        correct_predictions = 0
        total_predictions = 0
    
        # Wrap the dataset with tqdm for a progress bar
        progress_bar = tqdm(dataset, desc="Evaluating", unit=" example")
    
        with torch.no_grad():
            for example in progress_bar:
                prompt = f"<<Question>> {example['instruction']}\n <<Answer>>"
                expected_response = example['output'].strip()
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
                
                outputs = self.model.generate(input_ids, max_length=800, do_sample=False)
                generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if metric == 'accuracy':
                    total_predictions += 1
                    if generated_response.strip() == expected_response:
                        correct_predictions += 1
    
                results.append({
                    "prompt": prompt,
                    "expected_response": expected_response,
                    "generated_response": generated_response
                })
    
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
        with open(f'./data/{self.experiment_name}/evaluation_results.json', 'w') as outfile:
            json.dump({
                "results": results,
                "accuracy": accuracy
            }, outfile, indent=4)