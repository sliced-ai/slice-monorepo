from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
from datetime import datetime
from inference import InferenceEngine
from embedding import EmbeddingSystem
from auto_encoder import AutoEncoderTrainer
import torch
import random

app = Flask(__name__)
pipelines = {}

class ChatPipeline:
    def __init__(self, experiment_name, api_key):
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.history = []
        self.master_embeddings = []
        self.step = 0
        os.makedirs(f"data/{self.experiment_name}", exist_ok=True)
    
    def run_step(self, input_text, models_config, embedding_models_config, encoder_config):
        self.step += 1  # Increment step here
        step = self.step
        self.inference_engine = InferenceEngine(models_config)
        self.embedding_system = EmbeddingSystem(api_key=self.api_key, embedding_models_config=embedding_models_config)
        self.auto_encoder_trainer = AutoEncoderTrainer(encoder_config)
        
        step_data = {}
        response_texts = []
        raw_responses = self.inference_engine.generate_responses(input_text)
        for index, response in enumerate(raw_responses):
            data = self.inference_engine.extract_chat_completion_data(response)
            response_texts.append(data["Response Content"])
            
        step_data['responses'] = response_texts
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config, 'responses')
    
        embeddings = self.embedding_system.create_embeddings(response_texts)
        step_data['embeddings'] = embeddings
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config, 'embeddings')
    
        all_embeddings = [embed for model_embeds in embeddings.values() for embed in model_embeds]
        tensor_embeddings = [torch.tensor(embed, dtype=torch.float) for embed in all_embeddings]
        max_length = encoder_config.get('input_size', 5000)
        padded_embeddings = torch.stack([torch.cat([t, torch.zeros(max_length - t.size(0))]) if t.size(0) < max_length else t[:max_length] for t in tensor_embeddings])
    
        combined_embedding, autoencoder_encoded_embeddings, autoencoder_weights = self.auto_encoder_trainer.train_autoencoder(padded_embeddings)
        step_data['combined_embedding'] = combined_embedding
        step_data['autoencoder_weights'] = autoencoder_weights
    
        tsne_fig_path = f"data/{self.experiment_name}/step_{step}/embedding_visualization_tsne.png"
        grid_fig_path = f"data/{self.experiment_name}/step_{step}/embedding_visualization_3d.png"
        tsne_data_path = f"data/{self.experiment_name}/step_{step}/tsne_data.json"
        grid_data_path = f"data/{self.experiment_name}/step_{step}/grid_data.json"
        
        self.auto_encoder_trainer.visualize_embeddings_tsne(autoencoder_encoded_embeddings, tsne_fig_path, tsne_data_path)
        self.auto_encoder_trainer.visualize_2d_grid(autoencoder_encoded_embeddings, grid_fig_path, grid_data_path)
        
        step_data['tsne_fig_path'] = tsne_fig_path
        step_data['grid_fig_path'] = grid_fig_path
        step_data['grid_data_path'] = grid_data_path
        
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config, 'autoencoder')
    
        self.master_embeddings.append(combined_embedding)
        self.history.append(step_data)  # Append step data to history
    
        return random.choice(response_texts), tsne_fig_path, grid_fig_path, tsne_data_path, grid_data_path
    

    def save_intermediate_step_data(self, data, step, input_text, models_config, embedding_models_config, encoder_config, sub_step):
        step_folder = f"data/{self.experiment_name}/step_{step}"
        os.makedirs(step_folder, exist_ok=True)
    
        full_path = lambda filename: os.path.join(step_folder, filename)
        
        if sub_step == 'responses':
            with open(full_path("responses.json"), 'w') as f:
                json.dump(data['responses'], f, indent=4)
        
        if sub_step == 'embeddings':
            with open(full_path("embeddings.json"), 'w') as f:
                json.dump(data['embeddings'], f, indent=4)
        
        if sub_step == 'autoencoder':
        # Assuming files are saved directly to these paths from within their respective methods
            if 'tsne_fig_path' in data:
                assert os.path.exists(data['tsne_fig_path']), "TSNE file not found at expected location"
            if 'grid_fig_path' in data:
                assert os.path.exists(data['grid_fig_path']), "Grid visualization file not found at expected location"
        
        step_config = {
            'input_text': input_text,
            'models_config': models_config,
            'embedding_models_config': embedding_models_config,
            'encoder_config': encoder_config
        }
        with open(full_path("step_config.json"), 'w') as f:
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
        'input_size': 5000,
        'hidden_size': 512,
        'learning_rate': 0.001
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    experiment_name = request.form.get('experiment_name')
    input_text = request.form.get('input_text')
    api_key = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'
    use_default = 'use_default' in request.form

    if use_default:
        models_config = get_default_models_config()
        embedding_models_config = get_default_embedding_models_config()
        encoder_config = get_default_encoder_config()
    else:
        models_config = [{
            'name': request.form.get('model_name', 'gpt-4o'),
            'n': int(request.form.get('num_inferences', 1)),
            'max_tokens': int(request.form.get('max_tokens', 150)),
            'temperature': float(request.form.get('temperature', 0.7)),
            'top_p': float(request.form.get('top_p', 0.9))
        }]
        embedding_models_config = [{
            'name': request.form.get('embed_model_name', 'default_model'),
            'model': request.form.get('embed_model', 'text-embedding-3-large')
        }]
        encoder_config = {
            'input_size': int(request.form.get('input_size', 5000)),
            'hidden_size': int(request.form.get('hidden_size', 512)),
            'learning_rate': float(request.form.get('learning_rate', 0.001))
        }

    if experiment_name not in pipelines:
        pipelines[experiment_name] = ChatPipeline(experiment_name, api_key)
    chat_pipeline = pipelines[experiment_name]

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    chosen_response, tsne_fig_path, grid_fig_path, tsne_data_path, grid_data_path = chat_pipeline.run_step(input_text, models_config, embedding_models_config, encoder_config)
    
    data = {
        'chosen_response': chosen_response,
        'grid_fig_path': grid_fig_path,
        'message': 'Chat response generated!',
        'grid_data_path': grid_data_path,
        'tsne_data_path': tsne_data_path,
        'tsne_fig_path': tsne_fig_path,
        'responses': chat_pipeline.history[-1]['responses']  # Pass the actual responses
    }
    return jsonify(data)


@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
