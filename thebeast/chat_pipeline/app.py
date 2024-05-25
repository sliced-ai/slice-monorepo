
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from datetime import datetime
from inference import InferenceEngine
from embedding import EmbeddingSystem
from auto_encoder import AutoEncoderTrainer

app = Flask(__name__)

class ChatPipeline:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.history = []
        self.master_embeddings = []
        os.makedirs(f"data/{self.experiment_name}", exist_ok=True)

    def run_step(self, input_text, step, models_config, embedding_models_config, encoder_config):
        self.inference_engine = InferenceEngine(models_config)
        self.embedding_system = EmbeddingSystem(embedding_models_config)
        self.auto_encoder_trainer = AutoEncoderTrainer(encoder_config)
        
        step_data = {}
        responses = self.inference_engine.generate_responses(input_text)
        embeddings = self.embedding_system.create_embeddings(responses)
        combined_embedding, autoencoder_weights = self.auto_encoder_trainer.train_autoencoder(embeddings)
        
        step_data['responses'] = responses
        step_data['embeddings'] = embeddings
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
            f.write(data['autoencoder_weights'])
        
        # Save combined embedding
        with open(f"{step_folder}/combined_embedding.txt", 'w') as f:
            f.write(data['combined_embedding'])

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
            f.write("\n".join(self.master_embeddings))


def get_default_models_config():
    return [
        {'name': 'DefaultModel1', 'num_inferences': 2, 'hyper_params': {'param1': 'default1'}},
        {'name': 'DefaultModel2', 'num_inferences': 3, 'hyper_params': {'param2': 'default2'}}
    ]

def get_default_embedding_models_config():
    return [
        {'name': 'DefaultEmbedModel1', 'settings': {'setting1': 'default1'}},
        {'name': 'DefaultEmbedModel2', 'settings': {'setting2': 'default2'}}
    ]

def get_default_encoder_config():
    return {'encoder_type': 'default', 'settings': {'setting': 'default'}}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    experiment_name = request.form['experiment_name']
    use_default = 'use_default' in request.form

    if use_default:
        models_config = get_default_models_config()
        embedding_models_config = get_default_embedding_models_config()
        encoder_config = get_default_encoder_config()
    else:
        models_config = [{
            'name': request.form['model_name'],
            'num_inferences': int(request.form['num_inferences']),
            'hyper_params': parse_hyper_params(request.form['hyper_params'])
        }]
        
        embedding_models_config = [{
            'name': request.form['embed_model_name'],
            'settings': parse_embed_settings(request.form['embed_settings'])
        }]
        
        encoder_config = {
            'encoder_type': request.form['encoder_type'],
            'settings': parse_embed_settings(request.form['encoder_settings'])
        }

    chat_pipeline = ChatPipeline(experiment_name)
    input_text = request.form['input_text']
    step = 1
    chat_pipeline.run_step(input_text, step, models_config, embedding_models_config, encoder_config)
    
    return jsonify({"message": "Experiment started successfully!"})

def parse_hyper_params(hyper_params_str):
    params = hyper_params_str.split(',')
    return {p.split('=')[0]: p.split('=')[1] for p in params if '=' in p}

def parse_embed_settings(embed_settings_str):
    settings = embed_settings_str.split(',')
    return {s.split('=')[0]: s.split('=')[1] for s in settings if '=' in s}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)

