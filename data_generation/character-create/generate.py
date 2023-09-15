import openai
import os
import json
import boto3
import argparse
import logging
from typing import List, Dict, Any
from llama import Llama, Dialog  # Import Llama and Dialog from your custom library
from random_words import get_random_words
import re
import time
logging.basicConfig(level=logging.INFO)


class DataDownloader:
    def download_json_from_s3(self, s3_client, bucket: str, key: str, download_path: str) -> None:
        try:
            s3_client.download_file(bucket, key, download_path)
            logging.info(f"Successfully downloaded {key} from {bucket} to {download_path}")
        except Exception as e:
            logging.error(f"An error occurred while downloading the file from S3: {e}")
            raise

class DataProcessor:
    def load_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

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

class TextGenerator:
    def __init__(self, model_engine: str, api_key: str, use_llama: bool, ckpt_dir: str, tokenizer_path: str):
        self.model_engine = model_engine
        openai.api_key = api_key
        self.use_llama = use_llama
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.generator = None

    def generate_text(self, prompt: str, temperature: float = 0.85, max_tokens: int = 8192) -> str:
        if self.use_llama:
            if self.generator is None:
                # Code for Llama model
                self.generator = Llama.build(
                    ckpt_dir=self.ckpt_dir,
                    tokenizer_path=self.tokenizer_path,
                    max_seq_len=1500,
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



class FileManager:
    def save_to_file(self, text: str, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(text)


class MainController:
    def __init__(self, downloader: DataDownloader, processor: DataProcessor, text_generator: TextGenerator, file_manager: FileManager):
        self.downloader = downloader
        self.processor = processor
        self.text_generator = text_generator
        self.file_manager = file_manager

    def run(self, args) -> None:
        # Download JSON from S3
        s3_client = boto3.client('s3')
        self.downloader.download_json_from_s3(s3_client, args.bucket, args.key, args.download_path)

        # Load JSON data
        json_data = self.processor.load_json(args.download_path)

        # Generate core character using seed prompt
        """
        seed_prompt = json_data['Seed-Prompt']['text'].format(random_seed=get_random_words(4))
        core_character = self.text_generator.generate_text(seed_prompt, args.temperature, args.max_tokens)
        self.file_manager.save_to_file(processor.filter_character_info(core_character), 'core-character.txt')
        """

        
        # Start the timer
        start_time = time.time()

        n = 10  # Number of times to run inference
        with open("generated_texts.txt", "a") as f:  # Open a file to save the generated texts
            for i in range(n):
                seed_prompt = json_data['Seed-Prompt']['text'].format(random_seed=get_random_words(10))
                core_character = self.text_generator.generate_text(seed_prompt, args.temperature, args.max_tokens)
                f.write(processor.filter_character_info(core_character))  # Write the generated text to the file
                if i < n - 1:  # Don't add a delimiter after the last text
                    f.write("\n-----\n")  # Add a delimiter between texts
        core_character = processor.filter_character_info(core_character)
        #print(core_character)
        #print("\n\n")
        full_name = re.search(r'Full Name:\s+(.*?)\n', core_character)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Core Character time: {elapsed_time:.4f} seconds")
        
        # Process expansion prompts
        convo_details = []
        with open("expansion_data.txt", "a") as f: 
            for prompt_key, prompt_text in json_data['Expansion-Prompts'].items():
                formatted_prompt = prompt_text.format(random_seed=get_random_words(4), core_character=core_character)
                bullet_points = self.text_generator.generate_text(formatted_prompt, args.temperature, args.max_tokens)
                #print(f"\n\nExpansion Prompt {prompt_key}: {formatted_prompt}\n\n")
                pattern = r'^[•\d-]+\s+(.*)$'
                extracted_items = re.findall(pattern, bullet_points, re.MULTILINE)
                if not extracted_items:
                    pattern = r'^(.*?):\s+(.*)$'
                    extracted_items = re.findall(pattern, bullet_points, re.MULTILINE)
                convo_details += extracted_items
                f.write(str(extracted_items))  # Write the generated text to the file
                if i < n - 1:  # Don't add a delimiter after the last text
                    f.write("\n-----\n")  # Add a delimiter between texts
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Details time: {elapsed_time:.4f} seconds")        
        #print(convo_details)
        #print("\n\n")
        # Process conversation prompt
        with open("convo_data.txt", "a") as f: 
            for convo_detail_input in convo_details:
                convo_prompt = json_data['Conversation-Prompt']['text'].format(convo_detail_input=convo_detail_input,character_name=full_name,random_seed=get_random_words(4), core_character=core_character)
                #print(f"Conversation Prompt: {convo_prompt}")
                convo = self.text_generator.generate_text(convo_prompt, args.temperature, args.max_tokens)
                f.write(str(extracted_items))  # Write the generated text to the file
                if i < n - 1:  # Don't add a delimiter after the last text
                    f.write("\n-----\n")  # Add a delimiter between texts

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time: {elapsed_time:.4f} seconds")
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

    args = parser.parse_args()

    downloader = DataDownloader()
    processor = DataProcessor()
    text_generator = TextGenerator(args.model_engine, args.api_key, args.use_llama, args.ckpt_dir, args.tokenizer_path)
    file_manager = FileManager()
    main_controller = MainController(downloader, processor, text_generator, file_manager)

    main_controller.run(args)