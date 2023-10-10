import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


class FineTuner:

    def __init__(self, model_name, dataset_name, new_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model = new_model

    def _read_json_data(self, file_path):
        # This method reads and returns each conversation (list of dictionaries) from the jsonl file.
        conversations = []
        with open(file_path, 'r') as f:
            for line in f:
                conversations.append(json.loads(line.strip()))
        return conversations

    def _json_to_conversation(self, json_data):
        conversation_text = ""
        for entry in json_data:
            role = entry["role"].strip().replace(":", "")
            content = entry["content"].strip('"')
            conversation_text += f"### {role}: {content}\n"
        return {"text": conversation_text}


    def _transform_conversation(self, example):
        try:
            conversation_text = example['text']
            segments = conversation_text.split('###')
            cumulative_conversation = []
            reformatted_segments = []
            for i in range(0, len(segments) - 1, 2):
                role, human_text = segments[i].strip().split(":", 1)
                assistant_text = segments[i + 1].strip().replace('Input:', '').strip()
                if role.strip() == "Luna Rodriguez":
                    cumulative_conversation.append(f'<s>[INST]  [/INST] {human_text} </s>')
                else:
                    cumulative_conversation.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
                reformatted_segments.append(''.join(cumulative_conversation))
            return {'text': reformatted_segments}
        except Exception as e:
            print(f"Exception: {e}")


    def fine_tune(self,data_file):
        
        # Data Preparation
        conversations = self._read_json_data(data_file)
        transformed_conversations = [self._json_to_conversation(conversation) for conversation in conversations]
        flattened_conversations = [conv["text"] for conv in transformed_conversations]
        dataset = Dataset.from_dict({"text": flattened_conversations})
        transformed_dataset = dataset.map(self._transform_conversation)

        # Model Training
        training_arguments = TrainingArguments(
            output_dir="./results",
            num_train_epochs=4,
            per_device_train_batch_size=4,
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
    
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, "float16"),
                bnb_4bit_use_double_quant=False
            ),
            device_map={"": 0}
        )
    
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
        trainer = SFTTrainer(
            model=model,
            train_dataset=transformed_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )
    
        trainer.train()
    
        # Save the trained model
        trainer.model.save_pretrained(self.new_model)
    
        # Text Generation
        prompts = [
            "Hi what's your name? Can you tell me a bit about yourself?",
            "What's your favorite hobbies?"
        ]
        for prompt in prompts:
            result = pipeline(
                task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_length=200
            )(f"<s>[INST] {prompt} [/INST]")
            print(result[0]['generated_text'])

if __name__ == '__main__':
    tuner = FineTuner(
        model_name="NousResearch/Llama-2-7b-chat-hf",
        dataset_name="mlabonne/guanaco-llama2-1k",
        new_model="llama-2-7b-miniguanaco"
    )
    tuner.fine_tune("/home/ec2-user/environment/data_generation/cleaned_conversations.jsonl")
    
    
    
    
    
    
"""
        import pandas as pd
        from pprint import pprint
        from collections import Counter
        
        def visualize_dataset(dataset):
            # Option 1: Print the First Few Items
            print("\nOption 1: First Few Items\n")
            pprint(dataset[:5])
        
            # Option 2: Summarize the Dataset
            print("\nOption 2: Dataset Summary\n")
            print(f"Total entries: {len(dataset)}")
            if dataset:
                print(f"Keys in an entry: {list(dataset[0].keys())}")
        
            # Option 3: Utilize Pretty Printing
            print("\nOption 3: Pretty Printing\n")
            pprint(dataset[:5])
        
            # Option 4: Data Distribution (Assuming 'label' field for demonstration)
            if 'label' in dataset[0]:
                print("\nOption 4: Data Distribution\n")
                label_counts = Counter([entry['label'] for entry in dataset])
                print(label_counts)
        
            # Option 5: Visualize with Pandas
            print("\nOption 5: Visualization with Pandas\n")
            df = pd.DataFrame(dataset)
            print(df.head())
        
        # To use the function
        #visualize_dataset(transformed_dataset)

"""