from openai import OpenAI
import openai
import time
import multiprocessing
import random

API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

class EmbeddingSystem:
    def __init__(self, api_key, embedding_models_config):
        self.client = OpenAI(api_key=api_key)
        self.embedding_models_config = embedding_models_config

    def create_embeddings(self, texts):
        def worker(model_config, texts, return_dict, index):
            embeddings = []
            for text in texts:
                while True:
                    try:
                        embedding = self.get_embedding(text, model=model_config['model'])
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
            return_dict[index] = embeddings
            print(f"Process {index} completed with {len(texts)} texts.")
    
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
    
        total_texts = len(texts)
        max_texts_per_process = 5  # Adjust this number based on your needs
        num_processes = (total_texts + max_texts_per_process - 1) // max_texts_per_process
    
        print(f"Total texts to embed: {total_texts}")
        print(f"Running embedding with {num_processes} processes.")
    
        current_process = 0
        for model_config in self.embedding_models_config:
            for i in range(num_processes):
                start_index = i * max_texts_per_process
                end_index = min((i + 1) * max_texts_per_process, total_texts)
                texts_chunk = texts[start_index:end_index]
    
                p = multiprocessing.Process(target=worker, args=(model_config, texts_chunk, return_dict, current_process))
                jobs.append(p)
                p.start()
                current_process += 1
    
        for proc in jobs:
            proc.join()
    
        embeddings = {}
        for index, model_config in enumerate(self.embedding_models_config):
            model_name = model_config['name']
            model_embeddings = []
            for i in range(num_processes):
                if (index * num_processes + i) in return_dict:
                    model_embeddings.extend(return_dict[index * num_processes + i])
            embeddings[model_name] = model_embeddings
    
        print(f"Total embeddings generated: {sum(len(embeds) for embeds in embeddings.values())}")
        return embeddings

    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

def main():
    API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'
    embedding_models_config = [
        {'name': 'default_model', 'model': 'text-embedding-3-large'},
        {'name': 'small_model', 'model': 'text-embedding-3-small'}
    ]

    # Initialize the embedding system with API key and configurations
    embedding_system = EmbeddingSystem(api_key=API_KEY, embedding_models_config=embedding_models_config)

    # Sample texts to generate embeddings
    test_texts = ["Hello world!", "How are you doing today?", "Explain the theory of relativity.", "What is the capital of France?"]

    # Create embeddings
    all_embeddings = embedding_system.create_embeddings(test_texts)

    # Print results
    for model_name, embeddings in all_embeddings.items():
        print(f"Embeddings for {model_name}:")
        for i, embedding in enumerate(embeddings):
            print(f"Text: {test_texts[i]}, Embedding: {embedding}")

if __name__ == '__main__':
    main()
