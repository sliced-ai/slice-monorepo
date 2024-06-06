from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
from datetime import datetime
from inference import InferenceEngine
from embedding import EmbeddingSystem
from auto_encoder import AutoEncoderTrainer
import torch
import random
import umap.umap_ as umap
import matplotlib.pyplot as plt

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
    
    def run_step(self, input_text, models_config, embedding_models_config):
        self.step += 1
        step = self.step
        self.inference_engine = InferenceEngine(models_config)
        self.embedding_system = EmbeddingSystem(api_key=self.api_key, embedding_models_config=embedding_models_config)
        
        step_data = {}
        
        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Inferencing messages")
            
        raw_responses = self.inference_engine.generate_responses(input_text)
        response_data_list = self.inference_engine.extract_chat_completion_data(raw_responses)
        step_data = {
            'raw_responses': raw_responses,
            'uuid_responses': response_data_list
        }

        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, 'responses')

        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Embedding messages")
            
        embeddings = self.embedding_system.create_embeddings(response_data_list)
        step_data['embeddings'] = embeddings
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, 'embeddings')

        with open('/tmp/inference_status.txt', 'w') as f:
            f.write("Visualization")
        
        umap_fig_path = f"data/{self.experiment_name}/step_{step}/embedding_visualization_umap.png"
        umap_data_path = f"data/{self.experiment_name}/step_{step}/umap_data.json"
        
        # Use "text-embedding-3-large" embeddings for UMAP visualization
        large_embeddings = []
        for uuid, model_embeddings in embeddings.get('text-embedding-3-large', {}).items():
            large_embeddings.extend(model_embeddings)

        tensor_embeddings = [torch.tensor(embed, dtype=torch.float) for embed in large_embeddings]
        combined_embeddings = torch.stack(tensor_embeddings)

        self.visualize_embeddings_umap(combined_embeddings, umap_fig_path, umap_data_path)
        
        step_data['umap_fig_path'] = umap_fig_path
        
        self.save_intermediate_step_data(step_data, step, input_text, models_config, embedding_models_config, 'umap')
        
        self.master_embeddings.append(combined_embeddings)
        self.history.append(step_data)
        
        # Select a random response text to return
        chosen_response = random.choice(response_data_list)['response_content']
        
        return chosen_response, umap_fig_path, umap_data_path


    def save_intermediate_step_data(self, data, step, input_text, models_config, embedding_models_config, sub_step):
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
    
        elif sub_step == 'umap':
            if 'umap_fig_path' in data:
                assert os.path.exists(data['umap_fig_path']), "UMAP file not found at expected location"
    
        step_config = {
            'input_text': input_text,
            'models_config': models_config,
            'embedding_models_config': embedding_models_config
        }
        with open(full_path("step_config.json"), 'w') as f:
            json.dump(step_config, f, indent=4)


    def visualize_embeddings_umap(self, embeddings, umap_fig_path, json_path, title='2D Visualization of Embeddings'):
        n_neighbors = max(2, min(15, len(embeddings) - 1))  # Ensure n_neighbors is at least 2
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        try:
            umap_results = reducer.fit_transform(embeddings)
        except ValueError as e:
            print(f"UMAP error: {e}")
            # Handle potential zero-size array error or disconnected vertices
            umap_results = reducer.fit_transform(embeddings[:n_neighbors])  # Use a subset if necessary
    
        plt.figure(figsize=(10, 6))
        plt.scatter(umap_results[:, 0], umap_results[:, 1], alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig(umap_fig_path)
        plt.close()
    
        # Save the UMAP results to a JSON file
        umap_data = {'embeddings': umap_results.tolist()}
        with open(json_path, 'w') as f:
            json.dump(umap_data, f, indent=4)



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
                        uuid_response_file = os.path.join(step_dir, 'uuid_response_list.json')
                        if os.path.exists(uuid_response_file):
                            with open(uuid_response_file, 'r') as f:
                                responses = json.load(f)
                                token_count = sum(len(response['response_content'].split()) for response in responses)
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

    if experiment_name not in pipelines:
        pipelines[experiment_name] = ChatPipeline(experiment_name, api_key)
    chat_pipeline = pipelines[experiment_name]

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    chosen_response, umap_fig_path, umap_data_path = chat_pipeline.run_step(input_text, models_config, embedding_models_config)
    
    data = {
        'chosen_response': chosen_response,
        'message': 'Chat response generated!',
        'umap_data_path': umap_data_path,
        'umap_fig_path': umap_fig_path,
        'responses': chat_pipeline.history[-1]['uuid_responses']
    }
    return jsonify(data)

@app.route('/project_data/<project_name>', methods=['GET'])
def project_data(project_name):
    project_path = os.path.join('data', project_name)
    if not os.path.exists(project_path):
        return jsonify({"error": "Project not found"}), 404
    
    project_data = {'name': project_name, 'steps': []}
    for step in sorted(os.listdir(project_path)):
        step_path = os.path.join(project_path, step)
        if os.path.isdir(step_path):
            step_info = {'step': step}
            uuid_response_file = os.path.join(step_path, 'uuid_response_list.json')
            umap_data_file = os.path.join(step_path, 'umap_data.json')
            step_config_file = os.path.join(step_path, 'step_config.json')
            if os.path.exists(uuid_response_file):
                with open(uuid_response_file, 'r') as f:
                    step_info['responses'] = json.load(f)
            if os.path.exists(umap_data_file):
                with open(umap_data_file, 'r') as f:
                    step_info['umap_data'] = json.load(f)
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
