import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Constants
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, token="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")
    return tokenizer, model

def evaluate_model(model, tokenizer, prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    model.to(DEVICE)
    
    # Generate output using the model
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    base_prompt = "tell me a random number"
    reference_texts = ["Here is a random number: 42"]  # Example reference, replace with suitable reference texts
    
    tokenizer, model = load_tokenizer_and_model(MODEL_ID)
    
    # Perform inference
    generated_text = evaluate_model(model, tokenizer, base_prompt)
    print("Generated text:", generated_text)

    # Evaluate using BLEU score or other metrics
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=[generated_text], references=[reference_texts])
    print("BLEU score:", results['bleu'])

if __name__ == "__main__":
    main()
