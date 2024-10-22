import os
import json
from glob import glob
import torch
from transformers import PreTrainedTokenizerFast

def load_json_files(data_dir):
    file_paths = sorted(glob(os.path.join(data_dir, '*.json')))
    all_utterances = []
    for file_path in file_paths:
        with open(file_path, 'r') as data_file:
            data = json.load(data_file)
            for item in data:
                all_utterances.append(f"{item['name']}: {item['utterance']}")
    return all_utterances

def tokenize_and_save(utterances, tokenizer, output_path):
    # Concatenate all utterances into a single string
    full_text = "\n".join(utterances)
    # Tokenize the full text
    tokenized_data = tokenizer.encode(full_text, add_special_tokens=False)
    # Save the tokenized data
    torch.save(tokenized_data, output_path)

def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    return tokenizer

def main():
    data_dir = '/workspace/slice-monorepo/cl_cr3/aligneddata_final_cleaned'
    output_path = 'tokenized_utterances.pt'
    tokenizer_path = '/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json'
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Load and concatenate all text data
    all_utterances = load_json_files(data_dir)
    
    # Tokenize all utterances and save to a file
    tokenize_and_save(all_utterances, tokenizer, output_path)
    
    # Example data
    if all_utterances:
        example_text = all_utterances[0]
        example_tokens = tokenizer.encode(example_text, add_special_tokens=False)
        print(f"Example Text: {example_text}")
        print(f"Example Tokens: {example_tokens}")
        print(f"Decoded Tokens: {tokenizer.decode(example_tokens)}")
    
    print(f"Tokenized data saved to {output_path}. Total items: {len(all_utterances)}")

if __name__ == "__main__":
    main()
