import json
import os
import random
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import requests 

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'

LLM_SERVICE_URL = "http://localhost:5000"  # Replace with the actual LLM service URL
TEMPERATURE = 0.85
MAX_TOKENS = 7000

def start_service(ckpt_dir, tokenizer_path):
    data = {'ckpt_dir': ckpt_dir, 'tokenizer_path': tokenizer_path}
    response = requests.post(f"{LLM_SERVICE_URL}/start", json=data)
    print("Start Service:", response.text)

@app.route('/generate_data', methods=['POST'])
def generate_data():
    print("Request received:")
    data = request.json
    ckpt_dir = "/home/ec2-user/environment/pipeline/0_shared/models/llama-2-7b-chat"
    tokenizer_path = "/home/ec2-user/environment/pipeline/0_shared/models/tokenizer.model"
    
    # Start the service if not already started
    if 'service_started' not in session or not session['service_started']:
        start_service(ckpt_dir, tokenizer_path)
        session['service_started'] = True
    
    data = request.json
    input_text = data['input']
    
    # Generate character seed
    seed_prompt = f"Given the following paragraph, generate a detailed character description:\n\n{input_text}\n \n Generate a complete synthetic character description encompassing the following dimensions:\n- Full Name\n- Nickname (if any)\n- Age\n- Gender\n- Sexual Orientation\n- Ethnicity\n- Nationality\n- Religion\n- Height\n- Weight\n- Hair Color\n- Eye Color\n- Scars or Tattoos (if any)\n- Clothing Style\n- Openness\n- Conscientiousness\n- Extraversion\n- Agreeableness\n- Neuroticism\n- Smoking/Drinking (if any)\n- Hobbies\n- Favorite Food\n- Pet Peeves (if any)\n- Language Proficiency\n- Technical Skills (if any)\n- Social Skills\n- Other Skills (if any)\n- Place of Birth\n- Family (if any)\n- Education (if any)\n- Occupational History (if any)\n- Early life\n- Middle life\n- Current life\n- Significant Other (if any)\n- Friends (if any)\n- Enemies (if any)\n- Language & Vocabulary\n- Tone & Modulation\n- Non-verbal Cues\n- Motivations\n- Fears\n- Secrets\n\nPlease make the character as comprehensive and multi-dimensional as possible. My input is to not have this person be very technical or heavily knowledge-based. Basically, they cannot be a doctor or engineer.\nThe format for the output is a list of bullet points like the above input."
    
    # Send a POST request to generate the character seed
    generate_data = {
        "prompt": seed_prompt,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    generate_response = requests.post(f"{LLM_SERVICE_URL}/generate", json=generate_data)
    if generate_response.status_code == 200:
        character_seed = generate_response.json()["generated_text"]
    else:
        character_seed = "Failed to generate character seed."
    print(f"CHARACTER SEED: {character_seed}")
    return jsonify({'character_seed': character_seed})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    
    # Send a POST request to generate text
    generate_data = {
        "prompt": message,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    generate_response = requests.post(f"{LLM_SERVICE_URL}/generate", json=generate_data)
    if generate_response.status_code == 200:
        chat_response = generate_response.json()["generated_text"]
    else:
        chat_response = "Failed to generate response."
    
    return jsonify({'response': chat_response})

@app.route('/stop', methods=['POST'])
def stop_service():
    # Stop the service
    response = requests.post(f"{LLM_SERVICE_URL}/stop")
    print("Stop Service:", response.text)
    session['service_started'] = False
    return jsonify({'message': 'Service stopped successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)), debug=True)