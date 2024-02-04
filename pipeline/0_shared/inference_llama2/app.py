from flask import Flask, request, jsonify
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import os
import gc

app = Flask(__name__)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nDEVICE CHOICE IS: {device}\n\n")

def initialize_model(model_name, lora_params_path=None):
    global model, tokenizer
    # Load tokenizer with the same configuration as in fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model, potentially with LoRA parameters
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False
    ),
    device_map={"": 0},
    token = "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"
    )
    
    # Load LoRA parameters if specified
    if lora_params_path:
        load_lora_parameters(model, lora_params_path)
        print(f"\nLoRa Params: {lora_params_path}\n")
    model.eval()

def load_lora_parameters(model, load_path):
    # Attempt to load LoRA parameters similarly to the fine-tuning script
    try:
        lora_params = torch.load(os.path.join(load_path, 'lora_params.pt'), map_location=device)
        for name, param in model.named_parameters():
            if 'lora' in name and name in lora_params:
                param.data.copy_(lora_params[name])
        print("Loaded LoRA parameters successfully.")
    except FileNotFoundError:
        print("LoRA parameters file not found. Continuing with original parameters.")

@app.route('/start', methods=['POST'])
def start():
    content = request.json
    if 'model_name' in content:
        lora_params_path = content.get('lora_params_path')
        initialize_model(content['model_name'], lora_params_path)
        return jsonify({"message": "Model loaded successfully."}), 200
    else:
        return jsonify({"error": "Missing model_name in the request."}), 400

@app.route('/generate', methods=['POST'])
def generate():
    global model, tokenizer 
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not initialized. Call /start first."}), 400
    
    content = request.json
    if 'prompt' in content:
        prompt = content['prompt']
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate a response using the model
        #outputs = model.generate(input_ids, max_length=950, do_sample=True, top_k=50)
        outputs = model.generate(input_ids, max_length=950, do_sample=False)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"generated_text": generated_text}), 200
    else:
        return jsonify({"error": "No prompt provided."}), 400

@app.route('/stop', methods=['POST'])
def stop():
    global model, tokenizer 
    if model is not None:
        del model
        model = None
        gc.collect()  # Force a garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory cache
        return jsonify({"message": "Model and resources have been successfully released."}), 200
    else:
        return jsonify({"message": "Model is not running."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

