import json
import torch
import datetime
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
import argparse
from data_mover import DataMover
import os

import warnings
warnings.filterwarnings("ignore")


class FineTuner:

    def __init__(self, model_name, new_model, datamover):
        self.datamover = datamover
        self.model_name = model_name
        self.new_model = new_model
        
        with open("./data/meta.txt", 'r') as file:
            self.character_name = file.readline().strip()
            self.run_uuid = file.readline().strip()
        self.datamover.run_uuid = self.run_uuid
        
        print(f"\n\n\n########\n RUN ID: {self.run_uuid}\n NAME: {self.character_name}")
            
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
            pass
            #print(f"Exception: {e}")
    
    def _get_training_arguments(self):
        # Returns the training arguments
        return TrainingArguments(
            output_dir="./data",
            num_train_epochs=4,  # Adjust this for interactive training
            per_device_train_batch_size=8,
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
        # Returns the PEFT configuration
        return LoraConfig(
            lora_alpha=512,
            lora_dropout=0.1,
            r=256,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def _initialize_model(self):
        # Initializes and returns the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

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
        return model, tokenizer

    def interactive_conversation(self):
        # Initialize model and tokenizer
        model, tokenizer = self._initialize_model()

        print("Enter 'quit' to end the conversation.")
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_input = input(f"You ({timestamp}): ")
            if user_input.lower() == 'quit':
                break

            # Add timestamp to user input and generate response using the model
            timestamped_input = f"{timestamp} - {user_input}"
            response = self.generate_response(model, tokenizer, timestamped_input)
            response_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Model ({response_timestamp}): {response}")

            # Prepare data for training with timestamps
            training_data = {"input": timestamped_input, "response": f"{response_timestamp} - {response}"}
            self.train_on_interaction(model, tokenizer, training_data)

    def generate_response(self, model, tokenizer, timestamped_input):
        # Switch model to evaluation mode for response generation
        model.eval()

        # Generate response
        inputs = tokenizer.encode(f"<s>[INST] {timestamped_input} [/INST]", return_tensors='pt')
        outputs = model.generate(inputs, max_length=500)  # Adjust max_length as needed
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Switch back to training mode
        model.train()

        return response

    def train_on_interaction(self, model, tokenizer, interaction_data):
        # Create a list of dictionaries mimicking the structure expected by _json_to_conversation
        conversation_entries = [
            {"role": "User", "content": interaction_data['input']},
            {"role": "Model", "content": interaction_data['response']}
        ]

        # Convert interaction data to the format suitable for training
        transformed_conversation = self._json_to_conversation(conversation_entries)
        transformed_dataset = Dataset.from_dict({"text": [transformed_conversation["text"]]})

        # Create trainer and train the model using centralized configurations
        trainer = SFTTrainer(
            model=model,
            train_dataset=transformed_dataset,
            peft_config=self._get_peft_config(),
            dataset_text_field="text",
            max_seq_length=5000,
            tokenizer=tokenizer,
            args=self._get_training_arguments(),
            packing=False
        )
        trainer.train()

    def fine_tune(self,data_file):
        
        # Data Preparation
        conversations = self._read_json_data(data_file)
        transformed_conversations = [self._json_to_conversation(conversation) for conversation in conversations]
        flattened_conversations = [conv["text"] for conv in transformed_conversations]
        dataset = Dataset.from_dict({"text": flattened_conversations})
        transformed_dataset = dataset.map(self._transform_conversation)
        print(f"\nlen dataset: {dataset}")
        print(f"len transformed_dataset: {transformed_dataset}\n")

        model, tokenizer = self._initialize_model()

        trainer = SFTTrainer(
            model=model,
            train_dataset=transformed_dataset,
            peft_config=self._get_peft_config(),
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=tokenizer,
            args=self._get_training_arguments(),
            packing=False
        )
    
        trainer.train()
    
        # Save the trained model
        trainer.model.save_pretrained(self.new_model)
    
        # Text Generation
        prompts = [
        "Tell me about yourself",
        "What is your full name?",
        "Do you have a nickname? If so, what is it?",
        "How old are you?",
        "What is your gender?",
        "What is your sexual orientation?",
        "What is your ethnicity?",
        "What is your nationality?",
        "What is your religion, if any?",
        "How tall are you?",
        "What is your weight?",
        "What is your hair color?",
        "What is your eye color?",
        "Do you have any scars or tattoos? If so, please describe them.",
        "How would you describe your clothing style?",
        "How open are you to new experiences?",
        "Are you a conscientious person? Explain.",
        "Would you consider yourself extroverted or introverted?",
        "Are you generally agreeable with others?",
        "How would you describe your level of neuroticism?",
        "Do you smoke or drink? If so, how often?",
        "What are your hobbies or interests?",
        "What is your favorite food?",
        "Do you have any pet peeves?",
        "How proficient are you in languages? List the languages you know.",
        "Do you have any technical skills? If so, please specify.",
        "How would you rate your social skills?",
        "Do you have any other skills or talents? Please share.",
        "Where were you born?",
        "Tell me about your family.",
        "What is your educational background?",
        "Can you share some details about your occupational history?",
        "Describe your early life and upbringing.",
        "Tell me about your middle life experiences.",
        "What does your current life look like?",
        "Do you have a significant other? If so, tell me about them.",
        "Do you have close friends? Describe them.",
        "Have you ever had any enemies? If so, what led to that?",
        "How would you describe your language and vocabulary skills?",
        "Do you pay attention to your tone and modulation when communicating?",
        "Are you good at picking up non-verbal cues from others?",
        "What motivates you in life?",
        "Do you have any fears or phobias?",
        "Do you have any secrets you're willing to share?"
        ]
        
        #prompts = ["What is your full name?"]

        model_custom = "Hi "+self.character_name+" "

        prompt_response_list = []
        
        for prompt in prompts:
            result = pipeline(
                task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_length=1000
            )(f"<s>[INST] {model_custom}{prompt} [/INST]")
            
            prompt_response_pair = {
                'prompt': prompt,
                'response': result[0]['generated_text']
            }
            prompt_response_list.append(prompt_response_pair)
            
            file_path = os.path.join(datamover.log_folder, f"{datamover.run_uuid}_prompt_response.json")

            # Save the data to a JSON file
            with open(file_path, 'w') as f:
                json.dump(prompt_response_list, f, indent=4)
            
        self.datamover.logger.info(result[0]['generated_text'])    
        self.datamover.move_folder_to_s3("./data/", "slice-system-logs", "data_generation")

if __name__ == '__main__':
    
    datamover = DataMover(log_folder="./data")
    
    
    parser = argparse.ArgumentParser(description="Download a JSON file from S3 and generate text.")
    parser.add_argument('--convo_data', default="./data/cleaned_conversations.jsonl")
    parser.add_argument('--model_name', default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument('--new_model', default="./data/llama-2-7b-miniguanaco")
    parser.add_argument('--mode', choices=['fine_tune', 'conversation'], default='fine_tune', help='Select operation mode')
    args = parser.parse_args()

    tuner = FineTuner(
        model_name=args.model_name,
        new_model=args.new_model,
        datamover=datamover
    )

    if args.mode == 'fine_tune':
        datamover.logger.info("\n\n######## FINE TUNE MODEL\n\n")
        tuner.fine_tune(args.convo_data)
    elif args.mode == 'conversation':
        datamover.logger.info("\n\n######## Conversation MODEL\n\n")
        tuner.interactive_conversation()
    

    
    

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