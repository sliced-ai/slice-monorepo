import openai
import json
import re

def load_and_clean_file(filename):
    cleaned_lines = []
    with open(filename, 'r') as f:
        for line in f:
            # Remove leading numbers followed by a period and a space
            line = re.sub(r'^\d+\.\s+', '', line)
            # Remove leading dash and a space
            line = re.sub(r'^-\s+', '', line)
            # Remove leading and trailing whitespaces
            line = line.strip()
            cleaned_lines.append(line)
    return cleaned_lines
    
def load_convo_template(filename):
    with open(filename, 'r') as f:
        return f.read()

def format_for_llm(convo_template, detail_input):
    formatted_convo = convo_template.replace("{convo_detail_input}", detail_input)
    formatted_convo = formatted_convo.replace("{character_name}", "Sally")
    return formatted_convo

def infer_with_gpt4(messages, temperature=0.7, max_tokens=3500):
    openai.api_key = "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"
    model_engine = "gpt-3.5-turbo"  # Choose the GPT-4 engine you want to use

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response['choices'][0]['message']['content'].strip()

def save_for_later(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def reinsert_into_template(convo_template, detail_input):
    formatted_convo = format_for_llm(convo_template, detail_input)
    llm_input = [
        {
            "role": "user",
            "content": formatted_convo
        }
    ]
    return llm_input

if __name__ == "__main__":
    test_inference = False  # Set this flag to control whether to run inference

    # Step 1: Load the conversation template
    convo_template = load_convo_template('/home/ec2-user/environment/data_generation/datagen_prompts/convogen.txt')

    # Load dialogs from JSON files
    filename = '/home/ec2-user/environment/data_generation/data_expansion.txt'
    string_list = load_and_clean_file(filename)

    # Save only the detail_input for later inference
    save_for_later('detail_inputs.json', string_list)

    if test_inference:
        for detail_input in string_list:
            # Re-insert the detail_input into the convo_template and format for LLM
            llm_input = reinsert_into_template(convo_template, detail_input)

            # Perform inference
            response = infer_with_gpt4(llm_input)
            print(response)
            
"""
This will save the detail_input strings into a file called detail_inputs.json. The reinsert_into_template function takes a convo_template and a detail_input, and returns the formatted data that can be used for inference.

To use this saved data later, you can load it back in like so:
with open('detail_inputs.json', 'r') as f:
    saved_details = json.load(f)
Then, you can loop over saved_details and call reinsert_into_template(convo_template, detail) to prepare each one for inference.

"""