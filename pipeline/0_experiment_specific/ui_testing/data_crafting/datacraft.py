import json
import requests
import os
import random
import csv
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)
SERVICE_URL = "http://localhost:5000"
app.secret_key = 'your_secret_key'


Input_prompt= """
Respond to this input:
input: {input}
output:
"""
removeme = """
Respond to this input:"""


def start_service(model_name):
    data = {'model_name': model_name}
    response = requests.post(f"{SERVICE_URL}/start", json=data)
    print("Start Service:", response.text)

def generate_text(prompt, lora_params_path=None):
    data = {'prompt': prompt, 'lora_params_path': lora_params_path}
    response = requests.post(f"{SERVICE_URL}/generate", json=data)
    return response.json()

def stop_service():
    response = requests.post(f"{SERVICE_URL}/stop")
    print("Stop Service:", response.text)

def create_dataset_from_conversation(generated_text, dataset_path, min_char_length=5):
    # Split the generated_text by new lines to process each line individually
    lines = generated_text.split('\n')
    valid = False

    with open(dataset_path, 'a') as f:
        input_part, output_part = "", ""
        # Iterate through each line to find the one with "input:" and "output:"
        for line in lines:
            if line.startswith('input:'):
                input_part = line.replace("input:", "").strip()
            elif line.startswith('output:'):
                output_part = line.replace("output:", "").strip()

        # Check if both input and output meet the minimum character length requirement
        if len(input_part) >= min_char_length and len(output_part) >= min_char_length:
            valid = True
            # Adding "input:" and "output:" prefixes back for consistency
            formatted_input = f"input: {input_part}"
            formatted_output = f"output: {output_part}"
            # Prepare the data for saving. Using seq as 0 as per instructions.
            data = {
                "input": formatted_input,
                "output": formatted_output,
                "seq": 0  # Assigning sequence number as 0
            }
            # Write the JSON object to the file
            f.write(json.dumps(data) + '\n')

    return valid

    
def refine_generated_text(generated_text, original_prompt):
    lines = generated_text.split('\n')
    output_index = -1
    
    # Find the index of the last "output:" line
    for i, line in enumerate(lines):
        if line.startswith('output:'):
            output_index = i

    # If there's a line following "output:", move it to the same line
    if output_index != -1 and output_index + 1 < len(lines):
        lines[output_index] = lines[output_index] + " " + lines.pop(output_index + 1).strip()

    # Assuming the last two lines are the final input-output pair
    final_pair = lines[-2:] if len(lines) >= 2 else ""
    return '\n'.join(final_pair)
    
def recursive_tune(dataset_path, experiment_name, iterations, lora_params_path=None):
    data = {
        'dataset_path': dataset_path,
        'experiment_name': experiment_name,
        'iterations': iterations,
        'lora_params_path': lora_params_path,
    }
    response = requests.post(f"{SERVICE_URL}/recursive_tune", json=data)
    return response.json()

@app.route('/generate_text_datacraft', methods=['POST'])
def generate_text_route():
    print("Request received:")
    data = request.json
    input_text = data['paragraph']
    input_text = Input_prompt.format(input=input_text)
    dataset_folder = "/home/ec2-user/environment/ui_testing/data"
    dataset = "/home/ec2-user/environment/ui_testing/data/dataset.jsonl"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Start the service if not already started
    if 'service_started' not in session or not session['service_started']:
        start_service(model_name)
        session['service_started'] = True
        experiment_name = "dynamic_chat_1"
        lora_params_path = None

    # Append the new user input to the conversation history
    if 'conversation' not in session:
        session['conversation'] = []
    session['conversation'].append(f"User: {input_text}")

    # Combine the conversation history into a single prompt
    input_prompt = "\n".join(session['conversation'])
    # Call your existing Python code with the new prompt
    valid_conversation = False
    while valid_conversation == False:
        result = generate_text(input_prompt, "/home/ec2-user/environment/pipeline/0_shared/models/LoRa_paths/lora_params_100char.pt")
        generated_text = result.get("generated_text", "")
        print(f"GENERATED TEXT: {generated_text}")
        refined_text = refine_generated_text(generated_text, input_prompt)
        print(f"REFINED: {refined_text}\n\n")
        valid_conversation = create_dataset_from_conversation(refined_text,dataset)
        print(valid_conversation)

    stop_service()
    start_service(model_name)
    print(f"MY RESPONSE GENERATED: \n {refined_text} \n")    
    
    response = recursive_tune(dataset_folder, experiment_name, 1, lora_params_path)
    # Save the AI's response to the conversation history
    session['conversation'].append(f"AI: {generated_text}")
    
    # Return only the latest AI response to the front-end
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 4000)), debug=True)
