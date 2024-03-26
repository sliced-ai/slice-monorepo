import requests
import json

def main():
    # Set the URL of the Flask service
    url = "http://localhost:5000"

    # Start the model
    start_data = {
        "ckpt_dir": "/home/ec2-user/environment/pipeline/0_shared/models/llama-2-7b-chat",
        "tokenizer_path": "/home/ec2-user/environment/pipeline/0_shared/models/tokenizer.model"
    }
    start_response = requests.post(f"{url}/start", json=start_data)
    if start_response.status_code != 200:
        print("Failed to start the model.")
        return

    # Define the prompt for inference
    prompt = "What is the capital of France?"

    # Send a POST request to generate text
    generate_data = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100
    }
    generate_response = requests.post(f"{url}/generate", json=generate_data)
    if generate_response.status_code == 200:
        generated_text = generate_response.json()["generated_text"]
        print(f"Prompt: {prompt}")
        print(f"Generated Text: {generated_text}")
    else:
        print("Failed to generate text.")

    # Stop the model
    stop_response = requests.post(f"{url}/stop")
    if stop_response.status_code == 200:
        print("Model stopped successfully.")
    else:
        print("Failed to stop the model.")

if __name__ == "__main__":
    main()