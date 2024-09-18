import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
#MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer_and_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id,token="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").to(DEVICE)
    return tokenizer, model

def inference(model, tokenizer, base_prompt):
    inputs = tokenizer(base_prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_p=0.85
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("INFERENCE:")
    print("Generated text:", generated_text)

def training_inference_combined(model, tokenizer, input_text):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        print("Input IDs:", input_ids)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probabilities[0].topk(5)  # Look at the top 5 probabilities for more insight
        
        print("Logits Shape:", logits.shape)
        print("Logits:", logits)
        
        predicted_indices = logits[0].argmax(dim=-1)
        predicted_tokens = tokenizer.decode(predicted_indices, skip_special_tokens=True).replace('\n', ' ')
        
        print("Predicted Indices:", predicted_indices)
        print("Top 5 Probabilities and Indices at each position:")
        for i, (probs, indices) in enumerate(zip(top_probs, top_indices)):
            if i < 20:  # Increase the limit to see more positions
                print(f"Position {i}:")
                print("Probabilities:", probs.cpu().numpy())
                print("Indices:", indices.cpu().numpy())
                print("Tokens:", tokenizer.decode(indices))
                
        print("TRAINING INFERENCE:")
        print("Input text:", input_text)
        print("Predicted tokens:", predicted_tokens)

def main():
    base_prompt = "Hello, how are you today?"
    tokenizer, model = load_tokenizer_and_model(MODEL_ID)
    
    inference(model, tokenizer, base_prompt)
    training_inference_combined(model, tokenizer, base_prompt)

if __name__ == "__main__":
    main()
