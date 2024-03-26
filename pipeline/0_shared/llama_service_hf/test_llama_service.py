import requests

SERVICE_URL = "http://localhost:8080"

def start_service(model_name):
    data = {
        'model_name': model_name,
    }
    response = requests.post(f"{SERVICE_URL}/start", json=data)
    print("Start Service:", response.json())

def generate_text(prompt, lora_params_path=None):
    data = {'prompt': prompt}
    if lora_params_path:
        data['lora_params_path'] = lora_params_path
    response = requests.post(f"{SERVICE_URL}/generate", json=data)
    print("Generate Text:", response.json())

def stop_service():
    response = requests.post(f"{SERVICE_URL}/stop")
    print("Stop Service:", response.json())

def bulk_tune(dataset_path,lora_params_path,experiment_name):
    data = {
        'dataset_path': dataset_path,
        'lora_params_path': lora_params_path,
        'experiment_name': experiment_name,
    }
    response = requests.post(f"{SERVICE_URL}/bulk_tune", json=data)
    print("Finished Tuning:", response.json())

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model name
    #model_name = "NousResearch/Llama-2-7b-chat-hf"
    
    start_service(model_name)
    
    dataset_path = "/home/ec2-user/environment/pipeline/0_experiment_specific/v1_memory_validation_testing/data"
    lora_params_path = "/home/ec2-user/environment/pipeline/0_shared/llama_service/data/memorytesting_v1/lora_params.pt"
    experiment_name = "memorytesting_v1"
    
    #bulk_tune(dataset_path,lora_params_path,experiment_name)
    
    # Example prompts with and without LoRA parameters
    prompts = [
        {"prompt":  "02:12:2020 13:57:05 Hey there, Maxwell! Tell me about Sarah, who is she?"},
        #{"prompt":  "02:12:2010 13:57:05 Hey there, Maxwell! How's it going?", "lora_params_path": f"./data/{experiment_name}/lora_params.pt"},
        #{"prompt":  "02:12:2010 13:57:05 Hey there, Maxwell! Tell me about Sarah, who is she?", "lora_params_path": f"./data/{experiment_name}/lora_params.pt"},
        {"prompt":  "02:12:2020 13:57:05 Hey there, Maxwell! How's it going?"},
    ]
    
    for item in prompts:
        generate_text(item["prompt"], item.get("lora_params_path"))
        print("\n")

    stop_service()