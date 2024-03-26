import json
import os
import requests
import re
from quality_selector import analyze_and_select_best_text

SERVICE_URL = "http://localhost:8080"

def start_service(model_name):
    data = {'model_name': model_name}
    response = requests.post(f"{SERVICE_URL}/start", json=data)
    print("Start Service:", response.json())

def generate_text(prompt, lora_params_path=None):
    data = {'prompt': prompt}
    if lora_params_path:
        data['lora_params_path'] = lora_params_path
    response = requests.post(f"{SERVICE_URL}/generate", json=data)
    print("Generate Text:", response.json())
    return response.json()

def stop_service():
    response = requests.post(f"{SERVICE_URL}/stop")
    print("Stop Service:", response.json())

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    seed_file_path = "/home/ec2-user/environment/pipeline/0_prompts/character_prompts/dnd/Seed.1.txt"
    json_prompt_path = "/home/ec2-user/environment/pipeline/0_prompts/pipeline_v0_system_prompts/L0/demographics.json"

    # Debug output for paths
    #print(f"Seed file path: {seed_file_path}")
    #print(f"JSON prompt path: {json_prompt_path}")

    # Load seed text
    with open(seed_file_path, 'r') as seed_file:
        seed_text = seed_file.read().strip()
        #print(f"Seed text: {seed_text}")  # Debug output for seed text

    # Load and format JSON prompt
    with open(json_prompt_path, 'r') as json_file:
        prompt_data = json.load(json_file)
        formatted_prompt = prompt_data["Seed-Prompt"]["text"].format(seed=seed_text)
        #print(f"Formatted prompt: {formatted_prompt}\n")  # Debug output for formatted prompt

    start_service(model_name)
    
    sets = 10  # Number of outputs to generate
    output_file_path = "../data/seed_generation.txt"  # Path to the file where outputs will be saved
    
    with open(output_file_path, "a") as f:
        for i in range(sets):
            generated_response = generate_text(formatted_prompt)
            generated_text = generated_response["generated_text"]
            f.write(generated_text)
            if i < sets - 1:
                f.write("\n-----\n")
    
    # Analyze generated texts to select the best one
    best_text = analyze_and_select_best_text(output_file_path, "../data/analysis_result.json", "../data/analysis_chart.png")
    bullet_points = best_text
    # Assuming filter_character_info cleans up the best text for final use
    #final_text = extract_bullet_points(best_text)
    
    # Extract the full name from the best text
    match = re.search(r'Full Name:\s+(.*?)\n', best_text)
    if match:
        full_name = match.group(1)
    else:
        full_name = "Name not found"

    #print(f"\nGenerated text: {final_text}")  # Debug output for generated text

    #stop_service()

    #bullet_points = extract_bullet_points(generated_text)
    print(f"\n\nExtracted bullet points: {best_text}")  # Debug output for extracted bullet points

    def get_full_name_from_bullet_points(bullet_points):
        # Find the line with "Full Name:"
        for line in bullet_points.split('\n'):
            if "Full Name:" in line:
                # Extract the name part and remove leading/trailing spaces
                full_name = line.split("Full Name:", 1)[1].strip()
                return full_name
        return "Character_Unknown"  # Default name if not found
    
    # Extract the full name
    full_name = get_full_name_from_bullet_points(bullet_points)
    folder_name = full_name.replace(' ', '_')
    folder_path = os.path.join("/home/ec2-user/environment/pipeline/data", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    output_file_path = os.path.join(folder_path, "demographics.txt")
    with open(output_file_path, 'w') as output_file:
        output_file.write(bullet_points)

    print(f"Demographics saved to {output_file_path}")  # Confirmation of saved file
