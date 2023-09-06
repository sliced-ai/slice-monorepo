from typing import Optional, List
import json
import glob
import fire
from llama import Llama

def load_saved_details(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)

def reinsert_into_template(convo_template: str, detail_input: str) -> List[dict]:
    formatted_convo = convo_template.replace("{convo_detail_input}", detail_input)
    return [
        {
            "role": "user",
            "content": formatted_convo
        }
    ]

def load_dialogs_from_files(file_paths: List[str]) -> List[List[dict]]:
    all_dialogs = []
    for path in file_paths:
        with open(path, 'r') as f:
            dialogs = json.load(f)
        all_dialogs.extend(dialogs)
    return all_dialogs

def save_for_llm_training(result, filename):
    # Initialize the conversation list
    conversation = []

    # Split the content into lines and loop through each line
    for line in result['generation']['content'].split('\n'):
        # Include only the lines which start with character names or "Input:"
        if line.startswith("{") or line.startswith("Input:"):
            # Check if line contains '}', if not skip to the next line
            if '}' not in line:
                continue
            
            # Extract role and content
            role, content = line.split('}', 1)
            role = role[1:].strip()  # Removing the starting '{'
            content = content.strip()

            # Append to the conversation list
            conversation.append({
                'role': role,
                'content': content
            })

    # Save the conversation to a JSONL file
    with open(filename, 'a') as f:
        f.write(json.dumps(conversation) + '\n')


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.85,
    top_p: float = 0.7,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    convo_template_file: str = "path/to/your/convo_template.txt",
    saved_details_file: str = "path/to/your/detail_inputs.json"
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Load the saved details
    saved_details = load_saved_details(saved_details_file)

    # Load the conversation template
    with open(convo_template_file, 'r') as f:
        convo_template = f.read()

    for detail in saved_details:
        dialog = reinsert_into_template(convo_template, detail)
        result = generator.chat_completion(
            [dialog],  # Wrapping it in a list as the function might expect a list of dialogs
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]  # Assuming the function returns a list, we take the first (and only) element
        
        save_for_llm_training(result, 'llm_training_data.jsonl')

        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
    
    
    
if __name__ == "__main__":
    fire.Fire(main)
