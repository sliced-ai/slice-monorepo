import os
import json
import openai
import time
import random
import multiprocessing
import h5py
from openai import OpenAI

# Initialize OpenAI API key
API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'
openai.api_key = API_KEY

# Function to process a JSON file and extract the relevant data
def extract_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    
    extracted_data = []
    for document in data:
        for turn in document['TURNS']:
            for name in turn['NAMES']:
                extracted_data.append({
                    'name': name,
                    'utterance': ' '.join(turn['UTTERANCES']),
                    'turn_number': turn['NUMBER'],
                    'file_path': file_path
                })
    return extracted_data

# Wrapper function to enable multiprocessing
def extract_data_wrapper(args):
    return extract_data(*args)

# Load data from JSON files in the specified folder and its subdirectories using multiprocessing
def load_data(folder_path, num_workers=4):
    all_texts = []
    all_metadata = []
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_paths.append(os.path.join(root, file))

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(extract_data_wrapper, [(file_path,) for file_path in file_paths])

    for result in results:
        for item in result:
            all_texts.append(item['utterance'])
            all_metadata.append({
                'name': item['name'],
                'turn_number': item['turn_number'],
                'file_path': item['file_path']
            })

    print(f"Processed {len(file_paths)} files and extracted {len(all_texts)} utterances.")
    return all_texts, all_metadata

# Define the Embedding System
class EmbeddingSystem:
    def __init__(self, api_key, embedding_model):
        self.client = OpenAI(api_key=api_key)
        self.model = embedding_model

    def create_embeddings(self, texts, num_workers=4):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        total_texts = len(texts)
        chunk_size = 5  # Adjust this number based on your needs

        #print(f"Total texts to embed: {total_texts}")
        total_processes = (total_texts + chunk_size - 1) // chunk_size
        #print(f"Running embedding with {total_processes} processes.")

        current_process = 0
        for i in range(0, total_texts, chunk_size):
            texts_chunk = texts[i:i + chunk_size]

            p = multiprocessing.Process(target=self.worker, args=(texts_chunk, return_dict, current_process))
            jobs.append(p)
            p.start()
            current_process += 1

        for proc in jobs:
            proc.join()

        embeddings = []
        for result in return_dict.values():
            embeddings.extend(result)

        #print(f"Total embeddings generated: {len(embeddings)}")
        return embeddings

    def worker(self, texts, return_dict, index):
        embeddings = []
        for text in texts:
            while True:
                try:
                    embedding = self.get_embedding(text)
                    embeddings.append(embedding)
                    break
                except openai.RateLimitError:
                    wait_time = random.uniform(1, 60)  # Randomized wait time
                    #print(f"Rate limit hit. Process {index} waiting for {wait_time} seconds.")
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
        #print(f"Process {index} completed with {len(texts)} texts.")

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

def save_embeddings(embeddings, metadata, filename, model_name):
    with h5py.File(filename, 'a') as f:
        # Append new embeddings
        if 'embeddings' not in f:
            f.create_dataset('embeddings', data=embeddings, maxshape=(None, len(embeddings[0])))
            names_encoded = [meta['name'].encode('utf8') for meta in metadata]
            turn_numbers = [meta['turn_number'] for meta in metadata]
            file_paths_encoded = [meta['file_path'].encode('utf8') for meta in metadata]
            model_names_encoded = [model_name.encode('utf8') for _ in metadata]
            f.create_dataset('names', data=names_encoded, maxshape=(None,))
            f.create_dataset('turn_numbers', data=turn_numbers, maxshape=(None,))
            f.create_dataset('file_paths', data=file_paths_encoded, maxshape=(None,))
            f.create_dataset('model_names', data=model_names_encoded, maxshape=(None,))
        else:
            f['embeddings'].resize((f['embeddings'].shape[0] + len(embeddings)), axis=0)
            f['embeddings'][-len(embeddings):] = embeddings
            names_encoded = [meta['name'].encode('utf8') for meta in metadata]
            turn_numbers = [meta['turn_number'] for meta in metadata]
            file_paths_encoded = [meta['file_path'].encode('utf8') for meta in metadata]
            model_names_encoded = [model_name.encode('utf8') for _ in metadata]
            f['names'].resize((f['names'].shape[0] + len(names_encoded)), axis=0)
            f['names'][-len(names_encoded):] = names_encoded
            f['turn_numbers'].resize((f['turn_numbers'].shape[0] + len(turn_numbers)), axis=0)
            f['turn_numbers'][-len(turn_numbers):] = turn_numbers
            f['file_paths'].resize((f['file_paths'].shape[0] + len(file_paths_encoded)), axis=0)
            f['file_paths'][-len(file_paths_encoded):] = file_paths_encoded
            f['model_names'].resize((f['model_names'].shape[0] + len(model_names_encoded)), axis=0)
            f['model_names'][-len(model_names_encoded):] = model_names_encoded

    #print(f"Chunk processed and saved.")

# Worker function for multiprocessing
def embedding_worker(texts_chunk, metadata_chunk, model_config, api_key, chunk_index, output_file, lock):
    embedding_system = EmbeddingSystem(api_key=api_key, embedding_model=model_config['name'])
    embeddings = embedding_system.create_embeddings(texts_chunk)

    # Save embeddings for the chunk
    with lock:
        save_embeddings(embeddings, metadata_chunk, output_file, model_config['name'])

# Main function to execute the script
def main():
    folder_path = '/workspace/slice-monorepo/cl_cr3/aligneddata/c=3'
    embedding_model = 'text-embedding-3-small'  # Replace with the desired OpenAI model
    num_workers = 4
    output_file = 'utterance_embeddings.h5'
    
    # Load data
    print("Loading data...")
    texts, metadata = load_data(folder_path, num_workers=num_workers)
    total_texts = len(texts)
    chunk_size = 512  # Batch size for processing
    chunks = int(total_texts/chunk_size)
    # Initialize multiprocessing manager and lock
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    jobs = []

    # Divide the work into chunks
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, total_texts, chunk_size)]
    chunked_metadata = [metadata[i:i + chunk_size] for i in range(0, total_texts, chunk_size)]
    # Run a set number of workers at a time
    for chunk_index, (texts_chunk, metadata_chunk) in enumerate(zip(chunked_texts, chunked_metadata)):
        print(f"Chunk: {chunk_index}/{chunks}")
        p = multiprocessing.Process(target=embedding_worker, args=(texts_chunk, metadata_chunk, {'name': embedding_model}, API_KEY, chunk_index, output_file, lock))
        jobs.append(p)
        p.start()
        
        # Ensure no more than num_workers are running at once
        if len(jobs) >= num_workers:
            for job in jobs:
                job.join()
            jobs = []  # Reset jobs list for the next set of workers

    # Wait for remaining jobs to finish
    for job in jobs:
        job.join()
    
    print(f"All chunks processed and saved to {output_file}")

if __name__ == '__main__':
    main()