from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import gc
from fine_tune import FineTuner


app = Flask(__name__)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_global = None  # Keep the model name globally for re-use
lora_loaded = False  # Track if LoRA parameters are currently loaded

print(f"\n\nDEVICE: {device}\n\n")

def initialize_model(model_name):
    global model, tokenizer, model_name_global
    if model is None:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").half()
        model_name_global = model_name

def reset_model():
    global model, lora_loaded
    if lora_loaded:  # Reset model if LoRA was loaded
        initialize_model(model_name_global)
        lora_loaded = False
        print("\nRESET LoRa")

def load_lora_parameters(load_path):
    global model, lora_loaded
    lora_params = torch.load(load_path)
    for name, param in model.named_parameters():
        if 'lora' in name and name in lora_params:
            param.data = torch.from_numpy(lora_params[name])
    lora_loaded = True
    print(f"Loaded LoRA parameters successfully. FILE: {load_path}")

@app.route('/start', methods=['POST'])
def start():
    content = request.json
    if 'model_name' in content:
        initialize_model(content['model_name'])
        return jsonify({"message": "Model loaded successfully."}), 200
    else:
        return jsonify({"error": "Missing model_name in the request."}), 400

@app.route('/generate', methods=['POST'])
def generate():
    global model, tokenizer
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not initialized. Call /start first."}), 400
    
    content = request.json
    prompt = content.get('prompt')
    lora_params_path = content.get('lora_params_path')

    if lora_params_path:
        # Load new LoRA parameters if a different path is provided
        load_lora_parameters(lora_params_path)
    elif lora_loaded:
        # Reset the model to its original state if no LoRA path is provided and LoRA was previously loaded
        reset_model()
    
    model.eval()    

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids,
        max_length=2048,         # Maximum length of the output sequence
        temperature=0.7,        # Controls the randomness: lower = less random, higher = more random
        top_p=0.7,              # If set to float < 1, only the top p tokens with cumulative probability > top_p are kept for generation
        repetition_penalty=1.2, # The parameter for repetition penalty. 1.0 means no penalty. 
        early_stopping=False,   # Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
        pad_token_id=tokenizer.eos_token_id, # Padding token id
        do_sample=True,
        use_cache=True,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"generated_text": generated_text}), 200

@app.route('/stop', methods=['POST'])
def stop():
    global model, tokenizer
    if model is not None:
        del model
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({"message": "Model and resources have been successfully released."}), 200
    else:
        return jsonify({"message": "Model is not running."}), 400

@app.route('/bulk_tune', methods=['POST'])
def bulk_tune():
    global model, tokenizer, model_name_global
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not initialized. Call /start first."}), 400
    
    content = request.json
    prompt = content.get('prompt')
    dataset_path = content.get('dataset_path')
    lora_params_path = content.get('lora_params_path')
    experiment_name = content.get('experiment_name')
    
    if lora_params_path:
        # Load new LoRA parameters if a different path is provided
        load_lora_parameters(lora_params_path)
    
    tuner = FineTuner(model_name_global, experiment_name,model,tokenizer)
    # Paths to the dataset directories
    dataset_paths = {
        'train': f'{dataset_path}/train',
        'val': f'{dataset_path}/val'
    }
    
    quick_test_train = None
    quick_test_eval = 100
    
    # Train and infer
    tuner.train_and_infer(dataset_paths,quick_test_train)

    tuner.evaluate_and_save_scores(dataset_paths['val'],'accuracy',quick_test_eval)
    
    return jsonify({"message": f"{experiment_name} Model has been tuned"}), 200
    
@app.route('/recursive_tune', methods=['POST'])
def recursive_tune():
    global model, tokenizer, model_name_global, lora_loaded
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not initialized. Call /start first."}), 400
    
    content = request.json
    dataset_path = content['dataset_path']  # Assuming this is provided and valid
    experiment_name = content['experiment_name']
    iterations = content.get('iterations', 1)  # Default to 1 iteration if not specified
    
    lora_params_path = content.get('lora_params_path', None)  # Start with None or provided path

    for i in range(iterations):
        if lora_params_path:
            load_lora_parameters(lora_params_path)
        else:
            reset_model()
            initialize_model(model_name_global)

        model.train()
        tuner = FineTuner(model_name_global, experiment_name, model, tokenizer)
        
        # Adjusted to pass a dictionary instead of a string
        dataset_paths = {'train': dataset_path}  # Here is the adjustment

        tuner.train_and_infer(dataset_paths)  # Now passing a dictionary

        new_lora_params_path = f"{os.path.dirname(dataset_path)}/lora_params.pt"
        tuner.save_lora_parameters()
        
        lora_params_path = new_lora_params_path
        reset_model()

    return jsonify({"message": f"{experiment_name} Model has been recursively tuned for {iterations} iterations", "final_lora_params_path": lora_params_path}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
