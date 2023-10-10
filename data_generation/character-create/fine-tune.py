# Import all necessary libraries
import json
import os
import time
import re
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


class DataProcessingPipeline:
    def __init__(self, model_name, dataset_name, new_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model = new_model
        # ... Initialize other class-level variables ...

    def read_json_data(self, file_path):
        # Function to read JSON data from a given file path
        conversations = []
        with open(file_path, 'r') as f:
            for line in f:
                conversations.append(json.loads(line.strip()))
        return conversations

    def json_to_conversation(self, json_data):
        conversation_text = ""
        for entry in json_data:
            role = entry["role"].strip().replace(":", "")
            content = entry["content"].strip('"')
            conversation_text += f"### {role}: {content}\n"
        return {"text": conversation_text}

    def transform_conversation(self, example):
        try:
            conversation_text = example['text']
            segments = conversation_text.split('###')
            cumulative_conversation = []
            reformatted_segments = []
    
            for i in range(0, len(segments) - 1, 2):
                role, human_text = segments[i].strip().split(":", 1)
                assistant_text = segments[i + 1].strip().replace('Input:', '').strip()
    
                # Add the new segment to the cumulative conversation
                if role.strip() == "Sally Thompson":
                    cumulative_conversation.append(f'<s>[INST]  [/INST] {human_text} </s>')
                else:
                    cumulative_conversation.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
    
                # Add the entire cumulative conversation as a new data entry
                reformatted_segments.append(''.join(cumulative_conversation))
    
            return {'text': reformatted_segments}
    
        except Exception as e:
            #print(f"Error in example: {example}")
            print(f"Exception: {e}")

    def run_pipeline(self):
        # Step 1: Read JSON data
        conversations = self.read_json_data('/home/ec2-user/environment/data_generation/character-create/testdata.jsonl')
    
        # Step 2: Transform to conversations
        transformed_conversations = [self.json_to_conversation(conversation) for conversation in conversations]
        flattened_conversations = [conv["text"] for conv in transformed_conversations]
        dataset = Dataset.from_dict({"text": flattened_conversations})
        transformed_dataset = dataset.map(self.transform_conversation)
    
        # Step 3: Initialize Trainer and Train the Model
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
    
        # Save trained model
        trainer.model.save_pretrained(self.new_model)
    
        # Run text generation pipeline with our next model
        prompt = "Hi what's your name? Can you tell me a bit about yourself?"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])
        
        prompt = "What's your favorite hobbies?"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])

if __name__ == '__main__':
    pipeline_obj = DataProcessingPipeline(model_name="NousResearch/Llama-2-7b-chat-hf", 
                                          dataset_name="mlabonne/guanaco-llama2-1k",
                                          new_model="llama-2-7b-miniguanaco")
    pipeline_obj.run_pipeline()
