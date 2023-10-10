import openai
import boto3

import argparse
import logging
from typing import List, Dict, Any
import uuid
import re
import time
import os
import torch
import json
from datetime import datetime
logging.basicConfig(level=logging.INFO)


from quality_selector import analyze_and_select_best_text
from llama import Llama, Dialog 
from random_words import get_random_words
from fine_tuner import FineTuner
from extract_convo import ConversationProcessor

################################
class DataMover:
    
    ################################
    def __init__(self):
        generated_uuid = uuid.uuid4()
        uuid_string = str(generated_uuid)
        self.run_uuid = uuid_string
        
        print(f'\n####\nYOUR RUN ID IS : {uuid_string}\n####\n')

    ################################
    def download_json_from_s3(self, s3_client, bucket: str, key: str, download_path: str) -> None:
        try:
            s3_client.download_file(bucket, key, download_path)
            logging.info(f"Successfully downloaded {key} from {bucket} to {download_path}")
        except Exception as e:
            logging.error(f"An error occurred while downloading the file from S3: {e}")
            raise

    ################################
    def move_folder_to_s3(self, local_folder_path, bucket_name, s3_folder):
        """
        Moves a folder from a local path on an EC2 instance to a specified S3 bucket and folder.
        
        Parameters:
        - local_folder_path (str): The local path of the folder to be moved.
        - bucket_name (str): The name of the S3 bucket.
        - s3_folder (str): The folder in the S3 bucket where the folder will be moved to.
        """
        # Initialize a session using Amazon S3
        s3 = boto3.client('s3')
        
        # Iterate through each file in the local folder
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                # Construct the full local path of the current file
                local_file_path = os.path.join(root, file)
                
                # Construct the relative path of the current file within the local folder
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                
                # Construct the full S3 path
                s3_path = os.path.join(s3_folder, self.run_uuid, relative_path)
                
                try:
                    # Upload the file
                    s3.upload_file(local_file_path, bucket_name, s3_path)
                    print(f'Successfully moved {local_file_path} to {bucket_name}/{s3_path}')
                    
                    # Optionally, delete the local file after upload
                except Exception as e:
                    print(f'An error occurred: {e}')

################################
class DataProcessor:
    
    ################################
    def load_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    ################################
    def filter_character_info(self, text: str) -> str:
        # Define the list of bullet points you want to keep
        bullet_points = [
            "Full Name", "Nickname", "Age", "Gender", "Sexual Orientation",
            "Ethnicity", "Nationality", "Religion", "Height", "Weight",
            "Hair Color", "Eye Color", "Scars or Tattoos", "Clothing Style",
            "Openness", "Conscientiousness", "Extraversion", "Agreeableness",
            "Neuroticism", "Smoking/Drinking", "Hobbies", "Favorite Food",
            "Pet Peeves", "Language Proficiency", "Technical Skills",
            "Social Skills", "Other Skills", "Place of Birth", "Family",
            "Education", "Occupational History", "Early Life", "Middle Life",
            "Current Life", "Significant Other", "Friends", "Enemies",
            "Language & Vocabulary", "Tone & Modulation", "Non-verbal Cues",
            "Motivations", "Fears", "Secrets"
        ]
        
        filtered_info = ""
        
        # Loop through each bullet point to find and keep the corresponding text
        for point in bullet_points:
            match = re.search(f"{re.escape(point)}: ([^\n]*)", text)
            if match:
                filtered_info += f"{point}: {match.group(1)}\n"
        
        return filtered_info

    ################################
    def extract_bullet_points(self, text):
        # Regex patterns for bullet points
        patterns = [
            r'^\d+\.\s+(.*)',      # Pattern 1
            r'^\*\s+(.*)',         # Pattern 2
            r'^•\s+(.*)',          # Pattern 3
        ]
    
        # Check if any of the patterns match in the text
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
    
        return False  # None of the patterns matched
        
################################
class TextGenerator:
    
    ################################
    def __init__(self, model_engine: str, api_key: str, use_llama: bool, ckpt_dir: str, tokenizer_path: str):
        self.model_engine = model_engine
        openai.api_key = api_key
        self.use_llama = use_llama
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.generator = None
        
    ################################
    def generate_text(self, prompt: str, temperature: float = 0.85, max_tokens: int = 8192) -> str:
        if self.use_llama:
            if self.generator is None:
                # Code for Llama model
                self.generator = Llama.build(
                    ckpt_dir=self.ckpt_dir,
                    tokenizer_path=self.tokenizer_path,
                    max_seq_len=3000,
                    max_batch_size=1
                )
            dialogs = [
                [{"role": "user", "content": prompt}]
            ]
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=None,
                temperature=temperature,
                top_p=0.7 # high = deterministic
            )
            full_response = results[0]['generation']['content'].strip()
        else:
            # Code for GPT-3 model
            raw_response = openai.ChatCompletion.create(
                model=self.model_engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            full_response = raw_response['choices'][0]['message']['content'].strip()
        return full_response
    ################################
    def clear_gpu_memory(self):
        del self.generator
        self.generator = None
        torch.cuda.empty_cache()

    
################################
class FileManager:
    
    ################################
    def save_to_file(self, text: str, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(text)

################################
class MainController:
    
    ################################
    def __init__(self, downloader, processor, text_generator, file_manager,fast_run):
        self.fast_run = fast_run
        self.downloader = downloader
        self.processor = processor
        self.text_generator = text_generator
        self.file_manager = file_manager
        self.core_character = None
        self.full_name = None
        self.convo_details = None
        self.json_data = None
    
    ################################
    def generate_and_analyze_texts(self, args):
        if self.fast_run: 
            with open("./data/seed_generation.txt", "a") as f:
                seed_prompt = self.json_data['Seed-Prompt']['text'].format(random_seed=get_random_words(3))
                self.core_character = self.text_generator.generate_text(seed_prompt, args.temperature, args.max_tokens)
                f.write(self.processor.filter_character_info(self.core_character))
            self.core_character = self.processor.filter_character_info(self.core_character)
            self.full_name = re.search(r'Full Name:\s+(.*?)\n', self.core_character).group(1)
        else:
            with open("./data/seed_generation.txt", "a") as f:
                sets = 10
                for i in range(sets):
                    seed_prompt = self.json_data['Seed-Prompt']['text'].format(random_seed=get_random_words(3))
                    self.core_character = self.text_generator.generate_text(seed_prompt, args.temperature, args.max_tokens)
                    f.write(self.processor.filter_character_info(self.core_character))
                    if i < sets-1:
                        f.write("\n-----\n")
            best_text = analyze_and_select_best_text("data/generated_texts.txt", "data/analysis_result.json", "data/analysis_chart.png")
            self.core_character = self.processor.filter_character_info(best_text)
            self.full_name = re.search(r'Full Name:\s+(.*?)\n', self.core_character).group(1)
    
    ################################
    def generate_convo_details(self, args):
        if self.fast_run: 
            self.convo_details = []
            max_retries = 3  # You can adjust this based on your requirements
            
            with open("./data/expansion_generation.txt", "a") as f:
                for prompt_key, prompt_text in self.json_data['Expansion-Prompts'].items():
                    retries = 0
                    
                    while retries < max_retries:
                        rejected = 0
                        formatted_prompt = prompt_text.format(random_seed=get_random_words(4), core_character=self.core_character)
                        bullet_points = self.text_generator.generate_text(formatted_prompt, args.temperature, args.max_tokens)
                        clean_output = "Check to make sure the below is in bullet format and if not then format it.: \n"
                        extracted_items = self.text_generator.generate_text(clean_output + bullet_points, args.temperature, args.max_tokens)
                        extracted_list = extracted_items.splitlines()
                        
                        for bullet_point in extracted_list:
                            if self.processor.extract_bullet_points(bullet_point):
                                if len(bullet_point) > 15:
                                    self.convo_details.append(bullet_point)
                            else:
                                rejected += 1
                        
                        if rejected <= 2:
                            break  # Successful generation, so break out of the retry loop
                        
                        retries += 1  # Increment retry counter
                    
                    # If reached max_retries, you might want to handle it, e.g., logging that it exceeded max retries for this prompt
                    
                    f.write(str(self.convo_details))
                    f.write("\n-----\n")
                    break
        else:
            self.convo_details = []
            with open("./data/expansion_generation.txt", "a") as f:
                for prompt_key, prompt_text in self.json_data['Expansion-Prompts'].items():
                    formatted_prompt = prompt_text.format(random_seed=get_random_words(4), core_character=self.core_character)
                    bullet_points = self.text_generator.generate_text(formatted_prompt, args.temperature, args.max_tokens)
                    clean_output = "Check to make sure the below is in bullet format and if not then format it.: \n"
                    extracted_items = self.text_generator.generate_text(clean_output + bullet_points, args.temperature, args.max_tokens)
                    self.convo_details += extracted_items
                    f.write(str(extracted_items))
                    f.write("\n-----\n")

    ################################
    def generate_convo_data(self, args):
        if self.fast_run: 
            with open(f"./data/convo_data.jsonl", "a") as f:
                for convo_detail_input in self.convo_details:
                    convo_prompt = self.json_data['Conversation-Prompt']['text'].format(convo_detail_input=convo_detail_input, character_name=self.full_name, random_seed=get_random_words(4), core_character=self.core_character)
                    convo = self.text_generator.generate_text(convo_prompt, args.temperature, args.max_tokens)
                    clean_output = "Ideal Format:\nJSONL (JSON Lines format) where each line is a valid JSON object representing a single conversation. Each object should have at least these fields:\n\nrole: The name of the character speaking\ncontent: The text of what the character said\nExample:\njsonl\nCopy code\n{\"role\": \"Alice\", \"content\": \"Hello, how are you?\"}\n{\"role\": \"Bob\", \"content\": \"I'm good, thanks.\"}\n\nread this conversation below and format it correctly. Only include the conversation."
                    extracted_items = self.text_generator.generate_text(clean_output + convo, args.temperature, args.max_tokens)
                    
                    convo_json = {
                        "content": extracted_items.strip(),
                        "details": {
                            "convo_prompt": convo_detail_input,
                            "model" : "LLAMA7B-Chat",
                            "date" : str(datetime.now()),
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens
                        }
                    }
                    f.write(json.dumps(convo_json))
                    break
        else:  
            with open(f"./data/convo_data.txt", "a") as f:
                for convo_detail_input in self.convo_details:
                    convo_prompt = self.json_data['Conversation-Prompt']['text'].format(convo_detail_input=convo_detail_input, character_name=self.full_name, random_seed=get_random_words(4), core_character=self.core_character)
                    convo = self.text_generator.generate_text(convo_prompt, args.temperature, args.max_tokens)
                    clean_output = "Ideal Format:\nJSONL (JSON Lines format) where each line is a valid JSON object representing a single conversation. Each object should have at least these fields:\n\nrole: The name of the character speaking\ncontent: The text of what the character said\nExample:\njsonl\nCopy code\n{\"role\": \"Alice\", \"content\": \"Hello, how are you?\"}\n{\"role\": \"Bob\", \"content\": \"I'm good, thanks.\"}\n\nread this conversation below and format it correctly. Ignore everything except the conversation."
                    extracted_items = self.text_generator.generate_text(clean_output + convo, args.temperature, args.max_tokens)
                    convo_json = {
                        "content": extracted_items.strip(),
                        "details": {
                            "convo_prompt": convo_detail_input,
                            "model" : "LLAMA7B-Chat",
                            "date" : datetime.now(),
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens
                        }
                    }
                    f.write(json.dumps(convo_json))
                    f.write("\n-----\n")

    ################################
    def run(self, args):
        
        start_time = time.time()
        s3_client = boto3.client('s3')
        self.downloader.download_json_from_s3(s3_client, args.bucket, args.key, args.download_path)
        self.json_data = self.processor.load_json(args.download_path)

        logging.info("\n\n########   SEED GENERATION\n\n")
        self.generate_and_analyze_texts(args)
        #logging.info(f"\n\n########   BEST SEED: \n{self.core_character}\n\n\n")

        logging.info("\n\n########   EXPANSION GENERATION\n\n")
        self.generate_convo_details(args)
        #self.downloader.move_folder_to_s3(f"./data/", "slice-system-logs", "data_generation")
        
        logging.info("\n\n######## CONVO DATA GENERATION\n\n")
        self.generate_convo_data(args)
        
        # File paths for input and output files
        input_file = './data/convo_data.jsonl'
        #output_file = '/home/ec2-user/environment/data_generation/character-create/data/cleaned_conversations.jsonl'
        output_file = './data/cleaned_conversations.jsonl'
        # Create an instance of the class and run the methods
        processor = ConversationProcessor(input_file)
        processor.load_conversations()
        processor.clean_conversations()
        processor.save_to_jsonl(output_file)

        #self.downloader.move_folder_to_s3(f"./data/", "slice-system-logs", "data_generation")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"GENERATION TIME: {elapsed_time:.4f} seconds")
        
        start_time = time.time()
        logging.info("\n\n######## FINE TUNING\n\n")
        
        self.text_generator.clear_gpu_memory()
        tuner = FineTuner(
        model_name="NousResearch/Llama-2-7b-chat-hf",
        dataset_name="mlabonne/guanaco-llama2-1k",
        new_model="llama-2-7b-miniguanaco"
        )
        tuner.fine_tune(output_file)
        elapsed_time = end_time - start_time
        logging.info(f"FINE TUNING TIME: {elapsed_time:.4f} seconds")
        
################################
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Download a JSON file from S3 and generate text.")
    parser.add_argument('--bucket', required=True, help='Name of the S3 bucket')
    parser.add_argument('--key', required=True, help='Key of the file in the S3 bucket')
    parser.add_argument('--download_path', default='./downloaded_file.json', help='Path where the file will be downloaded')
    parser.add_argument('--model_engine', default='text-davinci-002', help='OpenAI model engine')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--temperature', type=float, default=0.85, help='Temperature setting for the model') # high = random
    parser.add_argument('--max_tokens', type=int, default=7000, help='Maximum number of tokens for the model output')
    parser.add_argument('--use_llama', action='store_true', help='Use Llama model')
    parser.add_argument('--ckpt_dir', default='', help='Checkpoint directory for Llama model')
    parser.add_argument('--tokenizer_path', default='', help='Tokenizer path for Llama model')
    parser.add_argument('--fast_run', default=False)

    args = parser.parse_args()
    fast_run = args.fast_run
    fast_run = True
    if fast_run: 
        logging.info("\n####\n\n#### FAST RUN IS ENABLED ####\n\n####\n")


    downloader = DataMover()
    processor = DataProcessor()
    text_generator = TextGenerator(args.model_engine, args.api_key, args.use_llama, args.ckpt_dir, args.tokenizer_path)
    file_manager = FileManager()
    main_controller = MainController(downloader, processor, text_generator, file_manager,fast_run)

    main_controller.run(args)
    