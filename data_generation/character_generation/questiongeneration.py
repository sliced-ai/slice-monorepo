import json

# Placeholder function for model inference. Replace this with your actual model inference code.
def model_inference(input_text):
    return f"Inferred question: What are your thoughts on '{input_text}'?"

def generate_and_save_questions(input_text, custom_sentence, json_file_path):
    # Step 1 & 2: Take custom text input and add a custom sentence to it
    modified_input = f"{input_text}\n{custom_sentence}"
    
    # Step 3: Inference that data
    inference_output = model_inference(modified_input)
    
    # Step 4: Takes the outputs of that inference and makes them the inputs for a new json file
    new_dialog = [
        {"role": "user", "content": inference_output}
    ]
    
    # Step 5: Append those outputs to the json file and save it for future inferencing
    try:
        with open(json_file_path, 'r') as f:
            existing_dialogs = json.load(f)
    except FileNotFoundError:
        existing_dialogs = []
    
    existing_dialogs.append(new_dialog)
    
    with open(json_file_path, 'w') as f:
        json.dump(existing_dialogs, f)

def read_form_data(json_file_path):
    # Read the JSON file containing form data
    try:
        with open(json_file_path, 'r') as f:
            form_data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file_path} not found.")
        return []
    
    # Extract the paragraph inputs into a list
    paragraph_list = [value for key, value in form_data.items()]
    
    return paragraph_list

# Example usage
json_file_path = "./inference/datageneration/forminput/form_data2.json"
paragraph_list = read_form_data(json_file_path)

print("Extracted Paragraphs:")
for i, paragraph in enumerate(paragraph_list):
    print(f"{i+1}. {paragraph}")
    input_text = paragraph
    custom_sentence = "This is a custom sentence."
    json_file_path = "./inference/datageneration/chatdata/questions_for_chat_completion.json"
    
    generate_and_save_questions(input_text, custom_sentence, json_file_path)
