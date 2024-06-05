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
        self.step += 1
        step = self.step
        self.inference_engine = InferenceEngine(models_config)
        self.embedding_system = EmbeddingSystem(api_key=self.api_key, embedding_models_config=embedding_models_config)
        self.auto_encoder_trainer = AutoEncoderTrainer(encoder_config)
        
        step_data = {}
        
        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Inferencing messages")
            
        raw_responses = self.inference_engine.generate_responses(input_text)
        response_data_list = self.inference_engine.extract_chat_completion_data(raw_responses)
        step_data = {
            'raw_responses': raw_responses,
            'uuid_responses': response_data_list
        }

        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config, 'responses')

        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Embedding messages")
            
        embeddings = self.embedding_system.create_embeddings(response_data_list)
        step_data['embeddings'] = embeddings
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, encoder_config, 'embeddings')

        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Training autoencoder")
        
        all_embeddings = [embed for model_embeds in embeddings.values() for embed in model_embeds]
        tensor_embeddings = [torch.tensor(embed, dtype=torch.float) for embed in all_embeddings]
        max_length = encoder_config.get('input_size', 5000)
        padded_embeddings = torch.stack([torch.cat([t, torch.zeros(max_length - t.size(0))]) if t.size(0) < max_length else t[:max_length] for t in tensor_embeddings])
            
        combined_embedding, autoencoder_encoded_embeddings, autoencoder_weights = self.auto_encoder_trainer.train_autoencoder(padded_embeddings)
        step_data['combined_embedding'] = combined_embedding
        step_data['autoencoder_weights'] = autoencoder_weights

        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Visualization")
        
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
        self.history.append(step_data)
        
        return random.choice(response_texts), tsne_fig_path, grid_fig_path, tsne_data_path, grid_data_path

    def save_intermediate_step_data(self, data, step, input_text, models_config, embedding_models_config, encoder_config, sub_step):
        step_folder = f"data/{self.experiment_name}/step_{step}"
        embed_folder = f"{step_folder}/embeddings"
        os.makedirs(step_folder, exist_ok=True)
        os.makedirs(embed_folder, exist_ok=True)
    
        full_path = lambda filename: os.path.join(step_folder, filename)
        full_path_embed = lambda filename: os.path.join(embed_folder, filename)
                
        if sub_step == 'responses':
                # Save the uuid_response_list as a JSON file
                uuid_response_file = full_path("uuid_response_list.json")
                with open(uuid_response_file, 'w') as f:
                    json.dump(data['uuid_responses'], f, indent=4)
        
                # Convert raw_responses to a serializable format
                serializable_raw_responses = []
                for response_data in data['raw_responses']:
                    serializable_data = {
                        'uuid': response_data['uuid'],
                        'response': serialize_chat_completion(response_data['response']),
                        'configuration': response_data['configuration']
                    }
                    serializable_raw_responses.append(serializable_data)
        
                # Save the raw_responses_list as a JSON file
                raw_responses_file = full_path("raw_responses_list.json")
                with open(raw_responses_file, 'w') as f:
                    json.dump(serializable_raw_responses, f, indent=4)
        

        elif sub_step == 'embeddings':
            embeddings_data = {}
            for model_name, model_embeddings in data['embeddings'].items():
                for uuid, embeds in model_embeddings.items():
                    if uuid not in embeddings_data:
                        embeddings_data[uuid] = {'uuid': uuid, 'embeddings': []}
                    for embed in embeds:
                        embeddings_data[uuid]['embeddings'].append({
                            'embedding': embed,
                            'model_name': model_name
                        })
            
            for uuid, embed_data in embeddings_data.items():
                embedding_file = full_path_embed(f"{uuid}.json")
                with open(embedding_file, 'w') as f:
                    json.dump(embed_data, f, indent=4)
    
        if sub_step == 'autoencoder':
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


def serialize_chat_completion(response):
    return {
        'id': response.id,
        'object': response.object,
        'created': response.created,
        'model': response.model,
        'usage': {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        },
        'choices': [
            {
                'message': choice.message.content if hasattr(choice.message, 'content') else "",
                'logprobs': {
                    'tokens': [token_logprob.token for token_logprob in choice.logprobs.content],
                    'logprobs': [token_logprob.logprob for token_logprob in choice.logprobs.content],
                    'top_logprobs': [
                        {top_logprob.token: top_logprob.logprob for top_logprob in token_logprob.top_logprobs}
                        for token_logprob in choice.logprobs.content
                    ]
                } if hasattr(choice, 'logprobs') and choice.logprobs else None
            } for choice in response.choices
        ]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/projects')
def projects():
    projects_data = []
    data_dir = 'data'
    if os.path.exists(data_dir):
        for experiment_name in os.listdir(data_dir):
            experiment_dir = os.path.join(data_dir, experiment_name)
            if os.path.isdir(experiment_dir):
                project_info = {'name': experiment_name, 'steps': 0, 'token_estimate': 0}
                for step in os.listdir(experiment_dir):
                    step_dir = os.path.join(experiment_dir, step)
                    if os.path.isdir(step_dir):
                        project_info['steps'] += 1
                        responses_file = os.path.join(step_dir, 'responses.json')
                        if os.path.exists(responses_file):
                            with open(responses_file, 'r') as f:
                                responses = json.load(f)
                                token_count = sum(len(response.split()) for response in responses)
                                project_info['token_estimate'] += token_count
                projects_data.append(project_info)
    return render_template('projects.html', projects=projects_data)

@app.route('/chat', methods=['POST'])
def chat():
    experiment_name = request.form.get('experiment_name')
    input_text = request.form.get('input_text')
    api_key = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

    models_config = [{
        'name': request.form.get('model_name', 'gpt-4o'),
        'n': int(request.form.get('num_inferences', 1)),
        'max_tokens_min': int(request.form.get('max_tokens_min', 50)),
        'max_tokens_max': int(request.form.get('max_tokens_max', 150)),
        'temperature_min': float(request.form.get('temperature_min', 0.5)),
        'temperature_max': float(request.form.get('temperature_max', 1.0)),
        'top_p_min': float(request.form.get('top_p_min', 0.5)),
        'top_p_max': float(request.form.get('top_p_max', 0.9))
    }]
    embedding_models_config = [{
        'model': request.form.get('embed_model_name', 'text-embedding-3-small,text-embedding-3-large')
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
        'responses': chat_pipeline.history[-1]['responses']
    }
    return jsonify(data)

@app.route('/project_data/<project_name>', methods=['GET'])
def project_data(project_name):
    project_path = os.path.join('data', project_name)
    if not os.path.exists(project_path):
        return jsonify({"error": "Project not found"}), 404
    
    project_data = {'name': project_name, 'steps': []}
    for step in os.listdir(project_path):
        step_path = os.path.join(project_path, step)
        if os.path.isdir(step_path):
            step_info = {'step': step}
            responses_file = os.path.join(step_path, 'responses.json')
            tsne_data_file = os.path.join(step_path, 'tsne_data.json')
            grid_data_file = os.path.join(step_path, 'grid_data.json')
            step_config_file = os.path.join(step_path, 'step_config.json')
            if os.path.exists(responses_file):
                with open(responses_file, 'r') as f:
                    step_info['responses'] = json.load(f)
            if os.path.exists(tsne_data_file):
                with open(tsne_data_file, 'r') as f:
                    step_info['tsne_data'] = json.load(f)
            if os.path.exists(grid_data_file):
                with open(grid_data_file, 'r') as f:
                    step_info['grid_data'] = json.load(f)
            if os.path.exists(step_config_file):
                with open(step_config_file, 'r') as f:
                    step_info['step_config'] = json.load(f)
            project_data['steps'].append(step_info)
    return jsonify(project_data)

@app.route('/status', methods=['GET'])
def status():
    if os.path.exists('/tmp/inference_status.txt'):
        with open('/tmp/inference_status.txt', 'r') as f:
            status = f.read()
    else:
        status = "No current status."

    return jsonify({'status': status})

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
