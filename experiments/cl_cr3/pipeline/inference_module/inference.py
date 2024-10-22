import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import random
import multiprocessing
import time
import os
import h5py
import uuid
import yaml
from torch.utils.data import DataLoader, Dataset
import openai

# Initialize OpenAI API key from config
def initialize_openai(api_key):
    openai.api_key = api_key
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]
        return self.tokenizer(item, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

class DistributedInference:
    def __init__(self, config, output_file):
        self.model_path = config['model_path']
        self.gpus = config['gpus']
        self.prompt = config['inference_settings']['prompt']
        self.total_inferences = config['inference_settings']['total_inferences']
        self.temperature_range = config['inference_settings']['temperature_range']
        self.top_p_range = config['inference_settings']['top_p_range']
        self.max_new_tokens_range = config['inference_settings']['max_new_tokens_range']
        self.embedding_model = config['embedding_settings']['embedding_model']
        self.output_file = output_file
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side=config['tokenizer_config']['padding_side'])
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.dataset = CustomDataset([self.prompt] * self.total_inferences, self.tokenizer, max_length=512)

    @staticmethod
    def run_inference(model_path, device, prompt, n_inferences, temperature_range, top_p_range, max_new_tokens_range, embedding_model, api_key, batch_size, dataset, tokenizer, queue):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            temperature = random.uniform(*temperature_range)
            top_p = random.uniform(*top_p_range)
            max_new_tokens = random.randint(*max_new_tokens_range)

            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, temperature=temperature, top_p=top_p, do_sample=True)

            for i in range(input_ids.size(0)):
                decoded_input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                decoded_output = tokenizer.decode(output[i], skip_special_tokens=True)

                model_response = decoded_output[len(decoded_input):].strip()
                unique_id = str(uuid.uuid4())

                metadata = {
                    "id": unique_id,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "generated_text": model_response,
                    "inference_model": model_path
                }

                queue.put((unique_id, metadata, model_response))

    @staticmethod
    def embed_and_save(queue, embedding_model, api_key, output_file):
        client = openai.OpenAI(api_key=api_key)
        with h5py.File(output_file, 'w') as f:
            f.attrs['embedding_model'] = embedding_model

            while True:
                item = queue.get()
                if item is None:
                    break

                unique_id, metadata, model_response = item
                embedding = client.embeddings.create(input=[model_response], model=embedding_model).data[0].embedding

                group = f.create_group(unique_id)
                group.attrs['temperature'] = metadata['temperature']
                group.attrs['top_p'] = metadata['top_p']
                group.attrs['max_new_tokens'] = metadata['max_new_tokens']
                group.attrs['generated_text'] = metadata['generated_text']
                group.attrs['inference_model'] = metadata['inference_model']
                group.attrs['embedding_model'] = embedding_model
                group.create_dataset('embedding', data=embedding)

    def run(self):
        multiprocessing.set_start_method('spawn', force=True)

        queue = multiprocessing.Queue()

        processes = []
        for gpu in self.gpus:
            p = multiprocessing.Process(target=DistributedInference.run_inference, args=(
                self.model_path, gpu, self.prompt, self.total_inferences // len(self.gpus), self.temperature_range, self.top_p_range, self.max_new_tokens_range, self.embedding_model, openai.api_key, 1, self.dataset, self.tokenizer, queue))
            processes.append(p)
            p.start()

        embed_process = multiprocessing.Process(target=DistributedInference.embed_and_save, args=(queue, self.embedding_model, openai.api_key, self.output_file))
        embed_process.start()

        for p in processes:
            p.join()

        queue.put(None)
        embed_process.join()

        self.print_sample_item()

    def print_sample_item(self):
        with h5py.File(self.output_file, 'r') as f:
            group_ids = list(f.keys())
            if group_ids:
                random_id = random.choice(group_ids)
                group = f[random_id]
                print("\nSample Item from HDF5 File:")
                print(f"UUID: {random_id}")
                print(f"Temperature: {group.attrs['temperature']}")
                print(f"Top P: {group.attrs['top_p']}")
                print(f"Max New Tokens: {group.attrs['max_new_tokens']}")
                print(f"Generated Text: {group.attrs['generated_text']}")
                print(f"Inference Model: {group.attrs['inference_model']}")
                print(f"Embedding Model: {group.attrs['embedding_model']}")
                print(f"Embedding: {group['embedding'][:]}\n")
            else:
                print("No responses found in the HDF5 file.")

from utils.logging import log_execution
from utils.retry import retry

@log_execution
@retry()
def run_inference(config, experiment_name, step):
    # Create the output directory if it doesn't exist
    experiment_folder = os.path.join("experiments", experiment_name, f"iteration_{step}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Set the output file path
    output_file = os.path.join(experiment_folder, config['embedding_settings']['output_file'])

    initialize_openai(config['embedding_settings']['api_key'])
    inference = DistributedInference(config, output_file)
    inference.run()
    return output_file

# Usage Example for testing
if __name__ == "__main__":
    import yaml

    # Load configuration from a YAML file
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Step 2: Inference and embedding
    experiment_name = "MyExperiment_20240701"
    step = 1
    cr3_data = None  # Placeholder, as cr3_data is not used in the provided code
    inference_embeddings_path = run_inference(config['inference_and_embedding'], experiment_name, step)
    print(f"Inference and embedding results saved to: {inference_embeddings_path}")
