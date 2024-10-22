import json
from datasets import Dataset

# Read the JSON Lines file and convert it into a list of Python dictionaries
conversations = []
with open('/home/ec2-user/environment/data_generation/character-create/testdata.jsonl', 'r') as f:
    for line in f:
        conversations.append(json.loads(line.strip()))

# Convert JSON data to a string of conversation with '###' separator
def json_to_conversation(json_data):
    conversation_text = ""
    for entry in json_data:
        role = entry["role"].strip().replace(":", "")
        content = entry["content"].strip('"')
        conversation_text += f"### {role}: {content}\n"
    return {"text": conversation_text}

# Convert the list of conversations to the required format
transformed_conversations = [json_to_conversation(conversation) for conversation in conversations]

# Flatten the list of dictionaries to be a list of strings
flattened_conversations = [conv["text"] for conv in transformed_conversations]

# Create a dataset from the transformed conversations
dataset = Dataset.from_dict({"text": flattened_conversations})

# Now you can use your existing transformation function
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments = []
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Sally Thompson:', '').strip()
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Input:', '').strip()
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

    return {'text': ''.join(reformatted_segments)}

# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)
