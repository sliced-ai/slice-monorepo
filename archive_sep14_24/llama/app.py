from flask import Flask, request, jsonify
import os
from llama import Llama, Dialog 
import torch
from train_evaluation import train_model, evaluate

app = Flask(__name__)
text_generator = None
model = None

class TextGenerator:
    
    ################################
    def __init__(self, ckpt_dir: str, tokenizer_path: str):
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.generator = None
        
    ################################
    def generate_text(self, prompt: str, temperature: float = 0.85, max_tokens: int = 8192) -> str:
        if self.generator is None:
            # Code for Llama model
            self.generator = Llama.build(
                ckpt_dir=self.ckpt_dir,
                tokenizer_path=self.tokenizer_path,
                max_seq_len=3000,
                max_batch_size=1
            )
        dialogs = [
            [{"role": "user", "content": prompt}]
        ]
        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=temperature,
            top_p=0.7 # high = deterministic
        )
        full_response = results[0]['generation']['content'].strip()

        return full_response
    ################################
    def clear_gpu_memory(self):
        del self.generator
        self.generator = None
        torch.cuda.empty_cache()


@app.route('/start', methods=['POST'])
def start():
    global text_generator
    content = request.json
    ckpt_dir = content.get('ckpt_dir')
    tokenizer_path = content.get('tokenizer_path')

    text_generator = TextGenerator(ckpt_dir, tokenizer_path)
    return jsonify({"message": "Model loaded successfully."}), 200

@app.route('/generate', methods=['POST'])
def generate():
    global text_generator
    if text_generator is None:
        return jsonify({"error": "Model not initialized. Call /start first."}), 400

    content = request.json
    prompt = content.get('prompt')
    temperature = content.get('temperature', 0.85)
    max_tokens = content.get('max_tokens', 8192)

    generated_text = text_generator.generate_text(prompt, temperature, max_tokens)
    return jsonify({"generated_text": generated_text}), 200

@app.route('/stop', methods=['POST'])
def stop():
    global text_generator
    if text_generator is not None:
        text_generator.clear_gpu_memory()
        text_generator = None
        return jsonify({"message": "Model and resources have been successfully released."}), 200
    else:
        return jsonify({"message": "Model is not running."}), 400

@app.route('/train', methods=['POST'])
def train_route():
    content = request.json
    train_dataset_path = content.get('train_dataset_path')
    eval_dataset_path = content.get('eval_dataset_path')

    model = train_model(train_dataset_path, eval_dataset_path)
    return jsonify({"message": "Model training completed successfully."}), 200

@app.route('/evaluate', methods=['POST'])
def evaluate_route():
    global model
    if model is None:
        return jsonify({"error": "Model not trained. Train the model first."}), 400

    content = request.json
    eval_dataset_path = content.get('eval_dataset_path')

    eval_dataset = load_dataset(eval_dataset_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=model.params.max_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loss = evaluate(model, eval_dataloader, device)
    return jsonify({"eval_loss": eval_loss}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)