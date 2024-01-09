import openai
import argparse
from typing import List, Dict, Any
import re
import time
import torch
import json
from datetime import datetime

from quality_selector import analyze_and_select_best_text
from llama import Llama, Dialog 
from random_words import get_random_words
from fine_tuner import FineTuner
from extract_convo import ConversationProcessor
from data_mover import DataMover

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
            "Full Name", "Nickname", "Age", "Gender", "Ethnicity", "Nationality",
            "Height", "Weight", "Hair Color", "Eye Color", "Scars or Tattoos",
            "Clothing Style", "Hobbies", "Favorite Food", "Language Proficiency",
            "Family", "Friends", "Education", "Occupational History", "Current Feelings and State",
            "Motivations", "Fears", "Secrets", "Past Year Overview", 
            "Early life experiences", "Basic education", "Playmates", "Simple hobbies",
            "Educational aspirations", "Developing social skills", "Complex emotional states",
            "Higher education", "Early career experiences", "Romantic relationships", 
            "Evolving personal beliefs", "Career progression", "Family dynamics", 
            "Maturing worldviews", "Life achievements", "Major life changes", 
            "Retirement or legacy thoughts"
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
    def __init__(self, datamover, processor, text_generator, file_manager,fast_run):
        self.fast_run = fast_run
        self.datamover = datamover
        self.processor = processor
        self.text_generator = text_generator
        self.file_manager = file_manager
        self.core_character = None
        self.full_name = None
        self.convo_details = None
        self.json_data = None
    
    ################################
    def generate_and_analyze_texts(self, args):
        if self.fast_run == True: 
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
            best_text = analyze_and_select_best_text("./data/seed_generation.txt", "data/analysis_result.json", "data/analysis_chart.png")
            self.core_character = self.processor.filter_character_info(best_text)
            self.full_name = re.search(r'Full Name:\s+(.*?)\n', self.core_character).group(1)
    
    ################################
    def generate_convo_details(self, args,current_age):
        if self.fast_run == True: 
            self.convo_details = []
            max_retries = 3  # You can adjust this based on your requirements
            
            with open("./data/expansion_generation.txt", "a") as f:
                for prompt_key, prompt_text in self.json_data['Expansion-Prompts'].items():
                    retries = 0
                    
                    while retries < max_retries:
                        rejected = 0
                        formatted_prompt = prompt_text.format(random_seed=get_random_words(4), core_character=self.core_character, age_range=current_age)
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
            max_retries = 3  # You can adjust this based on your requirements
            
            with open("./data/expansion_generation.txt", "a") as f:
                for prompt_key, prompt_text in self.json_data['Expansion-Prompts'].items():
                    retries = 0
                    
                    while retries < max_retries:
                        rejected = 0
                        formatted_prompt = prompt_text.format(random_seed=get_random_words(4), core_character=self.core_character, age_range=current_age)
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

    ################################
    def generate_convo_data(self, args):
        if self.fast_run == True: 
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
                    f.write("\n-----\n")
                    break
        else:  
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
                    f.write("\n-----\n")
        
        input_file = './data/convo_data.jsonl'
        output_file = './data/cleaned_conversations.jsonl'
        processor = ConversationProcessor(input_file)
        processor.load_conversations()
        processor.clean_conversations()
        processor.save_to_jsonl(output_file)

    ################################
    def year_summary_update(self,args,current_age):
        with open("./data/seed_generation.txt", "a") as f:
            summary_prompt = self.json_data['Year-Summary-Prompt']['text'].format(age_range = current_age,core_character = self.core_character)
            year_summary = self.text_generator.generate_text(summary_prompt, args.temperature, args.max_tokens)
            
            datamover.logger.info(f'Summary: \n\n{year_summary}')
            
            update_prompt = self.json_data['Seed-Update-Prompt']['text'].format(year_summary = year_summary, age_range = current_age,core_character = self.core_character)
            raw_core_character = self.text_generator.generate_text(update_prompt, args.temperature, args.max_tokens)
            
            self.core_character = self.processor.filter_character_info(raw_core_character)
            f.write(self.core_character)
        self.full_name = re.search(r'Full Name:\s+(.*?)\n', self.core_character).group(1)
        
    ################################
    def run(self, args):
        
        start_time = time.time()
        self.datamover.download_json_from_s3(args.bucket, args.key, args.download_path)
        self.json_data = self.processor.load_json(args.download_path)

        datamover.logger.info("\n\n########   SEED GENERATION\n\n")
        self.generate_and_analyze_texts(args)
        #datamover.logger.info(f"\n\n########   BEST SEED: \n{self.core_character}\n\n\n")

        max_age = 30
        for age in range(6,max_age+1):
            try:
                datamover.logger.info("\n\n########   AGE: "+str(age)+"\n\n")  
                datamover.logger.info("\n\n######## Year Update Summary\n\n")
                self.year_summary_update(args,age)
                datamover.logger.info("\n\n########   EXPANSION GENERATION\n\n")  
                self.generate_convo_details(args,age)
                datamover.logger.info("\n\n######## CONVO DATA GENERATION\n\n")
                self.generate_convo_data(args)
                
            except: #TODO MAKE THIS LESS DUMB
                datamover.logger.info("\n\n########   AGE: "+str(age)+"\n\n")  
                datamover.logger.info("\n\n######## Year Update Summary\n\n")
                self.year_summary_update(args,age)
                datamover.logger.info("\n\n########   EXPANSION GENERATION\n\n")  
                self.generate_convo_details(args,age)
                datamover.logger.info("\n\n######## CONVO DATA GENERATION\n\n")
                self.generate_convo_data(args)

        
        # Save some meta info to pass about easier
        filename = "./data/meta.txt"
        with open(filename, 'w') as file:
            file.write(self.full_name)
            file.write("\n")
            file.write(self.datamover.run_uuid)

        end_time = time.time()
        elapsed_time = end_time - start_time
        datamover.logger.info(f"GENERATION TIME: {elapsed_time:.4f} seconds")
        self.datamover.move_folder_to_s3(f"./data/", "slice-system-logs", "data_generation")

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
    parser.add_argument('--fast_run', default='false')
    
    args = parser.parse_args()  
    
    if args.fast_run == 'false':
        args.fast_run = False
    else:
        args.fast_run = True

    datamover = DataMover(log_folder="./data")
    processor = DataProcessor()
    text_generator = TextGenerator(args.model_engine, args.api_key, args.use_llama, args.ckpt_dir, args.tokenizer_path)
    file_manager = FileManager()
    main_controller = MainController(datamover, processor, text_generator, file_manager,args.fast_run)
    
    print(args.fast_run)
    if args.fast_run == True: 
        datamover.logger.info("\n####\n\n#### FAST RUN IS ENABLED ####\n\n####\n")

    main_controller.run(args)
    