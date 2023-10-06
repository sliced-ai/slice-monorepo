import json
import re

def analyze_convo_file(file_path, core_character_full_name):
    incomplete_responses = 0
    malformed_format = 0
    total_convos = 0

    with open(file_path, 'r') as f:
        all_lines = f.read().split("\n-----\n")

        for line in all_lines:
            if line.strip() == "":
                continue
            total_convos += 1
            try:
                convo_json = json.loads(line)
            except json.JSONDecodeError:
                malformed_format += 1
                continue
            
            if 'role' not in convo_json or 'content' not in convo_json:
                malformed_format += 1
                continue

            convo_text = convo_json['content']

            # Check if the character has incomplete responses
            incomplete_flags = re.findall(f"{re.escape(core_character_full_name)}: [^:]*\\w+[^:]*\\*$", convo_text)
            incomplete_responses += len(incomplete_flags)

    print(f"Total Conversations: {total_convos}")
    print(f"Malformed Formats: {malformed_format}")
    print(f"Incomplete Responses from {core_character_full_name}: {incomplete_responses}")

if __name__ == "__main__":
    core_character_details = "Full Name: Elara \"Ellie\" Thompson\n" # Replace with the actual content from your file
    full_name = re.search(r'Full Name:\s+(.*?)\n', core_character_details).group(1)
    analyze_convo_file(f"{full_name}_convo_data.txt", full_name)
