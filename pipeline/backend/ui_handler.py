import json
import os
import random
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from data_generation import DataProcessor, CharacterSeedGenerator
import requests 

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'

LLM_SERVICE_URL = "http://localhost:5000"  # Replace with the actual LLM service URL
TEMPERATURE = 0.85
MAX_TOKENS = 7000

processor = DataProcessor()
character_seed_generator = CharacterSeedGenerator(processor, LLM_SERVICE_URL)
"""
@app.route('/start', methods=['POST'])
def start_service():
    data = request.json
    model_name = data['model_name']
    # Simulating service start
    print(f"Starting service with model: {model_name}")
    time.sleep(1)  # Simulating some delay
    return jsonify({'message': 'Service started successfully'})
"""

def start_service(model_name):
    data = {'model_name': model_name}
    response = requests.post(f"{LLM_SERVICE_URL}/start", json=data)
    print("Start Service:", response.text)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    lora_params_path = data['lora_params_path']
    # Simulating text generation
    print(f"Generating text with prompt: {prompt}")
    time.sleep(1)  # Simulating some delay
    generated_text = f"Generated text for prompt: {prompt}"
    return jsonify({'generated_text': generated_text})

@app.route('/stop', methods=['POST'])
def stop_service():
    # Simulating service stop
    print("Stopping service")
    time.sleep(1)  # Simulating some delay
    return jsonify({'message': 'Service stopped successfully'})

@app.route('/recursive_tune', methods=['POST'])
def recursive_tune():
    data = request.json
    dataset_path = data['dataset_path']
    experiment_name = data['experiment_name']
    iterations = data['iterations']
    lora_params_path = data['lora_params_path']
    # Simulating recursive tuning
    print(f"Performing recursive tuning with dataset: {dataset_path}, experiment: {experiment_name}, iterations: {iterations}")
    time.sleep(1)  # Simulating some delay
    return jsonify({'message': 'Recursive tuning completed successfully'})

@app.route('/generate_data', methods=['POST'])
def generate_data():
    print("Request received:")
    data = request.json
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # Start the service if not already started
    if 'service_started' not in session or not session['service_started']:
        start_service(model_name)
        session['service_started'] = True
        lora_params_path = None
    
    data = request.json
    input_text = data['input']
    
    # Generate character seed
    seed_prompt = f"Given the following paragraph, generate a detailed character description:\n\n{input_text}\n \n Generate a complete synthetic character description encompassing the following dimensions:\n- Full Name\n- Nickname (if any)\n- Age\n- Gender\n- Sexual Orientation\n- Ethnicity\n- Nationality\n- Religion\n- Height\n- Weight\n- Hair Color\n- Eye Color\n- Scars or Tattoos (if any)\n- Clothing Style\n- Openness\n- Conscientiousness\n- Extraversion\n- Agreeableness\n- Neuroticism\n- Smoking/Drinking (if any)\n- Hobbies\n- Favorite Food\n- Pet Peeves (if any)\n- Language Proficiency\n- Technical Skills (if any)\n- Social Skills\n- Other Skills (if any)\n- Place of Birth\n- Family (if any)\n- Education (if any)\n- Occupational History (if any)\n- Early life\n- Middle life\n- Current life\n- Significant Other (if any)\n- Friends (if any)\n- Enemies (if any)\n- Language & Vocabulary\n- Tone & Modulation\n- Non-verbal Cues\n- Motivations\n- Fears\n- Secrets\n\nPlease make the character as comprehensive and multi-dimensional as possible. My input is to not have this person be very technical or heavily knowledge-based. Basically, they cannot be a doctor or engineer.\nThe format for the output is a list of bullet points like the above input."
    #seed_prompt = input_text
    character_seed = character_seed_generator.generate_character_seed(seed_prompt, TEMPERATURE, MAX_TOKENS)
    
    # Simulating data generation
    print(f"Generating data for input: {input_text}")
    time.sleep(1)  # Simulating some delay
    generated_data = f"Generated data for input: {input_text}"
    
    return jsonify({'generated_data': generated_data, 'character_seed': character_seed})

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    edited_data = data['data']
    # Simulating model training
    print(f"Training model with edited data: {edited_data}")
    time.sleep(1)  # Simulating some delay
    training_progress = random.randint(0, 100)
    return jsonify({'training_progress': training_progress})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    # Simulating chat response
    print(f"Generating chat response for message: {message}")
    time.sleep(1)  # Simulating some delay
    chat_response = f"AI response to: {message}"
    return jsonify({'response': chat_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)