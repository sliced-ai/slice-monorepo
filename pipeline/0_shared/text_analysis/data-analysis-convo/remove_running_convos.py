import json

def filter_conversations(jsonl_file_path, output_file_path, skip_chars=20):
    # Open the original file and the output file
    with open(jsonl_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Process each line (which corresponds to a full conversation)
        for line in input_file:
            conversation = json.loads(line.strip())
            # Check if any message content after skipping initial characters contains a colon
            if any(':' in message['content'][skip_chars:] for message in conversation):
                continue  # Skip this conversation
            # If no colon is found after skipping the initial characters, write the conversation to the output file
            output_file.write(json.dumps(conversation) + '\n')

# Example usage
input_file_path = '/home/ec2-user/environment/model_training/Fine_Tune/sim_ns_role_duplicate_11724_dated.jsonl'  # Update with the actual path
output_file_path = '/home/ec2-user/environment/model_training/Fine_Tune/sim_ns_role_run_duplicate_11724_dated.jsonl'  # Update with the desired output path

filter_conversations(input_file_path, output_file_path)