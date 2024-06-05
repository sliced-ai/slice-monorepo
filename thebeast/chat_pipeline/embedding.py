API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

import json
import os
import openai
import random
import time
import multiprocessing
from openai import OpenAI

class EmbeddingSystem:
    def __init__(self, api_key, embedding_models_config):
        self.client = OpenAI(api_key=api_key)
        self.embedding_models_config = self.parse_models_config(embedding_models_config)

    def parse_models_config(self, embedding_models_config):
        parsed_config = []
        for config in embedding_models_config:
            models = config['model'].split(',')
            for model in models:
                parsed_config.append({'name': model.strip(), 'model': model.strip()})
        return parsed_config

    def create_embeddings(self, data):
        def worker(model_config, texts, uuids, return_dict, index):
            model_name = model_config['name']
            model = model_config['model']
            embeddings = []
            for text in texts:
                while True:
                    try:
                        embedding = self.get_embedding(text, model)
                        embeddings.append(embedding)
                        break
                    except openai.RateLimitError:
                        wait_time = random.uniform(1, 120)  # Randomized wait time
                        print(f"Rate limit hit. Process {index} waiting for {wait_time} seconds.")
                        time.sleep(wait_time)
                    except openai.APIError as e:
                        print(f"OpenAI API returned an API Error: {e}")
                        break
                    except openai.APIConnectionError as e:
                        print(f"Failed to connect to OpenAI API: {e}")
                        time.sleep(2)
                    except Exception as e:
                        print(f"Unexpected error in process {index}: {e}")
                        break
            return_dict[index] = {"uuids": uuids, "embeddings": embeddings, "model": model_name}
            print(f"Process {index} completed with {len(texts)} texts.")

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        total_texts = len(data)
        max_texts_per_process = 5  # Adjust this number based on your needs

        print(f"Total texts to embed: {total_texts}")
        total_processes = len(self.embedding_models_config) * ((total_texts + max_texts_per_process - 1) // max_texts_per_process)
        print(f"Running embedding with {total_processes} processes.")

        current_process = 0
        for model_config in self.embedding_models_config:
            for i in range(0, total_texts, max_texts_per_process):
                texts_chunk = [item['response_content'] for item in data[i:i + max_texts_per_process]]
                uuids_chunk = [item['uuid'] for item in data[i:i + max_texts_per_process]]

                p = multiprocessing.Process(target=worker, args=(model_config, texts_chunk, uuids_chunk, return_dict, current_process))
                jobs.append(p)
                p.start()
                current_process += 1

        for proc in jobs:
            proc.join()

        embeddings = {}
        for model_config in self.embedding_models_config:
            model_name = model_config['name']
            embeddings[model_name] = {}

        for result in return_dict.values():
            model_name = result['model']
            uuids = result['uuids']
            embeds = result['embeddings']
            for uuid, embed in zip(uuids, embeds):
                if uuid not in embeddings[model_name]:
                    embeddings[model_name][uuid] = []
                embeddings[model_name][uuid].append(embed)

        print(f"Total embeddings generated: {sum(len(embeds) for model_embeds in embeddings.values() for embeds in model_embeds.values())}")
        return embeddings

    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding


def main():
    API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

    # Sample input configuration
    embedding_models_config = [{
        'model': 'text-embedding-3-large,text-embedding-3-small'
    }]

    # Initialize the embedding system with API key and configurations
    embedding_system = EmbeddingSystem(api_key=API_KEY, embedding_models_config=embedding_models_config)

    # Sample texts to generate embeddings
    response_data_list = [
        {'uuid': '8b98ec5d-07f6-44cc-8dc6-acec733691e8', 'response_content': "The biggest cat in the world is the Siberian tiger, also known as the Amur tiger (Panthera tigris altaica). Male Siberian tigers can weigh up to 600 pounds (approximately 272 kilograms) and measure up to 10.5 feet (about 3.2 meters) in length, including their tail."},
        {'uuid': '6687fe6f-3c41-4cb4-b404-c4904e58dade', 'response_content': "The biggest cat in the world is the Siberian tiger, also known as the Amur tiger (Panthera tigris altaica). Males can weigh up to 660 pounds (300 kilograms) and measure up to 10.5 feet (3.3 meters) in length, including the tail. They are known for their powerful build and thick fur, which helps them survive in the cold climates of their natural habitat in the Russian Far East, as well as parts of China and North Korea."}
    ]

    # Create embeddings
    all_embeddings = embedding_system.create_embeddings(response_data_list)

    # Print results
    for model_name, embeddings in all_embeddings.items():
        print(f"Embeddings for {model_name}:")
        for uuid, embeds in embeddings.items():
            for embed in embeds:
                print(f"UUID: {uuid}, Embedding: ")

if __name__ == '__main__':
    main()
