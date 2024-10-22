import os
import uuid
import json
import torch
import random
import multiprocessing
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CONFIGS = {
    "bofenghuang/vigogne-2-7b-instruct": {"batch_size": 1},  # 7B parameters
    "bofenghuang/vigogne-7b-instruct": {"batch_size": 1},  # 7B parameters
    "SebastianSchramm/Cerebras-GPT-111M-instruction": {"batch_size": 1},  # 111M parameters
    "dominguesm/Canarim-7B-Instruct": {"batch_size": 1},  # 7B parameters
    # "PartAI/Dorna-Llama3-8B-Instruct": {"batch_size": 1},  # 8B parameters, Gated Repo
    "MaziyarPanahi/vigostral-7b-chat-Mistral-7B-Instruct-v0.1-GGUF": {"batch_size": 1},  # 7B parameters
    "philschmid/instruct-igel-001": {"batch_size": 1},  # Not specified, assuming < 10B
    "TheBloke/Vigogne-2-7B-Instruct-GGUF": {"batch_size": 1},  # 7B parameters
    "AlirezaF138/Dorna-Llama3-8B-Instruct-GGUF": {"batch_size": 1},  # 8B parameters
    "MaziyarPanahi/vigostral-7b-chat-Mistral-7B-Instruct-v0.1": {"batch_size": 1},  # 7B parameters
    "TheBloke/Vigogne-2-7B-Instruct-GPTQ": {"batch_size": 1},  # 7B parameters
    "bofenghuang/vigogne-mpt-7b-instruct": {"batch_size": 1},  # 7B parameters
    "bofenghuang/vigogne-falcon-7b-instruct": {"batch_size": 1},  # 7B parameters
    "bofenghuang/vigogne-bloom-7b1-instruct": {"batch_size": 1},  # 7B parameters
    "bofenghuang/vigogne-opt-6.7b-instruct": {"batch_size": 1},  # 6.7B parameters
    "abdullahalzubaer/bloom-6b4-clp-german-instruct-lora-peft": {"batch_size": 1}  # 6.4B parameters
    # Commented out due to CUDA memory errors
    # "bofenghuang/vigogne-2-7b-instruct": {"batch_size": 1},  # 7B parameters
    # "bofenghuang/vigogne-7b-instruct": {"batch_size": 1},  # 7B parameters
    # "AlirezaF138/Dorna-Llama3-8B-Instruct-GGUF": {"batch_size": 1},  # 8B parameters
    # "SebastianSchramm/Cerebras-GPT-111M-instruction": {"batch_size": 1}  # 111M parameters
}



GPUS = [0, 1, 2, 3, 4, 5]  # List of GPU indices available
PROMPT = "What's the biggest cat in the world?"
INFERENCES_PER_MODEL = 200  # Total number of inferences per model
TEMPERATURE_RANGE = (0.1, 2.0)
TOP_P_RANGE = (0.1, 1.0)
TOKEN_LENGTH_RANGE = (100, 250)
RESULTS_FILE = "inference_results.json"

# Function to create a unique configuration for each inference
def create_inference_config():
    temperature = random.uniform(*TEMPERATURE_RANGE)
    top_p = random.uniform(*TOP_P_RANGE)
    token_length = random.randint(*TOKEN_LENGTH_RANGE)
    return {
        "temperature": temperature,
        "top_p": top_p,
        "token_length": token_length,
        "do_sample": True
    }

# Function to load model and tokenizer
def load_model(model_name, device):
    print(f"Loading model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

# Function to run inference
def run_inference(model, tokenizer, config, device, batch_size):
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
    input_ids = input_ids.repeat(batch_size, 1)  # Create batch
    attention_mask = torch.ones(input_ids.shape, device=device)  # Set attention mask

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=config["token_length"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        do_sample=config["do_sample"],
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
    )
    responses = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(batch_size)]
    return responses

# Worker function to process inference tasks
def worker(gpu_index, status_dict, results_dict, queue, gpu_model_map):
    device = torch.device(f"cuda:{gpu_index}")
    while not queue.empty():
        model_name = queue.get()
        gpu_model_map[gpu_index] = model_name
        tokenizer, model = load_model(model_name, device)
        while status_dict[model_name]["waiting"] > 0:
            try:
                config = create_inference_config()
                status_dict[model_name]["in_progress"] += 1
                batch_size = MODEL_CONFIGS[model_name]["batch_size"]
                responses = run_inference(model, tokenizer, config, device, batch_size)
                for response in responses:
                    result = {
                        "uuid": str(uuid.uuid4()),
                        "model": model_name,
                        "config": config,
                        "response": response
                    }
                    results_dict[model_name].append(result)
                status_dict[model_name]["completed"] += batch_size
            except Exception as e:
                status_dict[model_name]["failed"] += batch_size
                error_message = str(e)
                with open(f"errors_{model_name}.log", "a") as error_file:
                    error_file.write(f"{uuid.uuid4()}: {error_message}\n")
            finally:
                status_dict[model_name]["in_progress"] -= 1
                status_dict[model_name]["waiting"] -= batch_size
                if status_dict[model_name]["waiting"] <= 0:
                    save_model_results(model_name, results_dict[model_name])
                    results_dict[model_name] = []  # Clear results after saving

# Save results to a single file at the end of each model's run
def save_model_results(model_name, results):
    with open(RESULTS_FILE, "a") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

# Function to print status updates
def print_status(status_dict, gpu_model_map):
    while any(v['waiting'] > 0 or v['in_progress'] > 0 for v in status_dict.values()):
        os.system('cls' if os.name == 'nt' else 'clear')
        for model, status in status_dict.items():
            print(f"Model: {model}")
            print(f"  Completed: {status['completed']}")
            print(f"  In Progress: {status['in_progress']}")
            print(f"  Failed: {status['failed']}")
            print(f"  Waiting: {status['waiting']}")
        print("\nGPU Utilization:")
        for gpu_index in GPUS:
            model = gpu_model_map[gpu_index]
            print(f"  GPU {gpu_index}: Model {model}")
        time.sleep(5)

# Prepare inference tasks
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    results_dict = manager.dict({model: manager.list() for model in MODEL_CONFIGS})
    status_dict = manager.dict({model: manager.dict({"completed": 0, "in_progress": 0, "waiting": INFERENCES_PER_MODEL, "failed": 0}) for model in MODEL_CONFIGS})
    gpu_model_map = manager.dict({gpu_index: None for gpu_index in GPUS})
    model_queue = manager.Queue()

    for model_name in MODEL_CONFIGS.keys():
        model_queue.put(model_name)

    # Preload models to avoid blocking inference
    preload_processes = []
    for gpu_index, model_name in zip(GPUS, MODEL_CONFIGS.keys()):
        gpu_model_map[gpu_index] = model_name
        p = multiprocessing.Process(target=load_model, args=(model_name, torch.device(f"cuda:{gpu_index}")))
        p.start()
        preload_processes.append(p)

    # Wait for all models to be preloaded
    for p in preload_processes:
        p.join()

    # Start worker processes for each GPU
    processes = []
    for gpu_index in GPUS:
        p = multiprocessing.Process(target=worker, args=(gpu_index, status_dict, results_dict, model_queue, gpu_model_map))
        p.start()
        processes.append(p)

    # Start status printing process
    status_process = multiprocessing.Process(target=print_status, args=(status_dict, gpu_model_map))
    status_process.start()

    # Wait for all tasks to complete
    for p in processes:
        p.join()

    # Ensure status process terminates
    status_process.terminate()
    status_process.join()

    print("Inference complete. Results saved to inference_results.json")
