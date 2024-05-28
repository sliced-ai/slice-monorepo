from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from datetime import datetime
from inference import InferenceEngine
from embedding import EmbeddingSystem
from auto_encoder import AutoEncoderTrainer
import torch

app = Flask(__name__)

class ChatPipeline:
    def __init__(self, experiment_name, api_key):
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.history = []
        self.master_embeddings = []
        os.makedirs(f"data/{self.experiment_name}", exist_ok=True)

    def run_step(self, input_text, step, models_config, embedding_models_config, encoder_config):
        self.inference_engine = InferenceEngine(models_config)
        self.embedding_system = EmbeddingSystem(api_key=self.api_key, embedding_models_config=embedding_models_config)
        self.auto_encoder_trainer = AutoEncoderTrainer(encoder_config)
        
        step_data = {}
        responses = self.inference_engine.generate_responses(input_text)
        
        # Ensure responses are a list of strings
        response_texts = [response['text'] if isinstance(response, dict) and 'text' in response else str(response) for response in responses]
        
        embeddings = self.embedding_system.create_embeddings(response_texts)
        
        step_data['responses'] = responses
        step_data['embeddings'] = embeddings
        
        # Flatten the embeddings from all models into a single list
        all_embeddings = [embed for model_embeds in embeddings.values() for embed in model_embeds]
        
        # Convert list of embeddings to PyTorch tensors and pad/truncate as necessary
        tensor_embeddings = [torch.tensor(embed, dtype=torch.float) for embed in all_embeddings]
        max_length = encoder_config.get('input_size', 5000)  # Default to some value if not specified
        padded_embeddings = torch.stack([torch.cat([t, torch.zeros(max_length - t.size(0))]) if t.size(0) < max_length else t[:max_length] for t in tensor_embeddings])
        
        combined_embedding, autoencoder_weights = self.auto_encoder_trainer.train_autoencoder(padded_embeddings)
        
        step_data['combined_embedding'] = combined_embedding
        step_data['autoencoder_weights'] = autoencoder_weights
        
        self.save_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config)
        
        self.master_embeddings.append(combined_embedding)
        
        return combined_embedding

    def save_step_data(self, data, step, input_text, models_config, embedding_models_config, encoder_config):
        step_folder = f"data/{self.experiment_name}/step_{step}"
        os.makedirs(step_folder, exist_ok=True)
        
        # Save responses
        with open(f"{step_folder}/responses.json", 'w') as f:
            json.dump(data['responses'], f, indent=4)
        
        # Save embeddings
        with open(f"{step_folder}/embeddings.json", 'w') as f:
            json.dump(data['embeddings'], f, indent=4)
        
        # Save autoencoder weights
        with open(f"{step_folder}/autoencoder_weights.txt", 'w') as f:
            f.write(str(data['autoencoder_weights']))
        
        # Save combined embedding
        with open(f"{step_folder}/combined_embedding.txt", 'w') as f:
            f.write(str(data['combined_embedding']))

        # Save step configuration
        step_config = {
            'input_text': input_text,
            'models_config': models_config,
            'embedding_models_config': embedding_models_config,
            'encoder_config': encoder_config
        }
        with open(f"{step_folder}/step_config.json", 'w') as f:
            json.dump(step_config, f, indent=4)

    def save_master_embedding(self):
        master_file = f"data/{self.experiment_name}/master_embedding.txt"
        with open(master_file, 'w') as f:
            f.write("\n".join(map(str, self.master_embeddings)))


def get_default_models_config():
    return [
        {"name": "gpt-4o", "n": 1, "max_tokens": 150, "temperature": 0.7, "top_p": 0.9},
        {"name": "gpt-4o", "n": 1, "max_tokens": 200, "temperature": 0.6, "top_p": 1.0}
    ]

def get_default_embedding_models_config():
    return [
        {'name': 'default_model', 'model': 'text-embedding-3-large'},
        {'name': 'small_model', 'model': 'text-embedding-3-small'}
    ]

def get_default_encoder_config():
    return {
        'input_size': 1000,
        'hidden_size': 512,
        'learning_rate': 0.001
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    experiment_name = request.form['experiment_name']
    #api_key = request.form['api_key']
    api_key = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'
    use_default = 'use_default' in request.form

    if use_default:
        models_config = get_default_models_config()
        embedding_models_config = get_default_embedding_models_config()
        encoder_config = get_default_encoder_config()
    else:
        models_config = [{
            'name': request.form['model_name'],
            'n': int(request.form['num_inferences']),
            'max_tokens': int(request.form['max_tokens']),
            'temperature': float(request.form['temperature']),
            'top_p': float(request.form['top_p'])
        }]
        
        embedding_models_config = [{
            'name': request.form['embed_model_name'],
            'model': request.form['embed_model']
        }]
        
        encoder_config = {
            'input_size': int(request.form['input_size']),
            'hidden_size': int(request.form['hidden_size']),
            'learning_rate': float(request.form['learning_rate'])
        }

    chat_pipeline = ChatPipeline(experiment_name, api_key)
    input_text = request.form['input_text']
    step = 1
    chat_pipeline.run_step(input_text, step, models_config, embedding_models_config, encoder_config)
    
    return jsonify({"message": "Experiment started successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
