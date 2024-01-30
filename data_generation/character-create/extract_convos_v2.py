import json
import re
from collections import Counter

class ConversationProcessor:
    
    def __init__(self, filename, error_log_file):
        self.filename = filename
        self.error_log_file = error_log_file
        self.original_conversation_count = 0
        self.cleaned_conversation_count = 0
        self.conversations = []
        self.conversations_cleaned = []
        self.error_counter = Counter()

    def load_conversations(self):
        with open(self.filename, 'r') as file, open(self.error_log_file, 'w') as error_log:
            sections = file.read().split('-----')
            for section in sections:
                section = self.fix_json_format(section)
                try:
                    data = json.loads(section)
                    conversation_content = data.get("content", None)
                    if conversation_content:
                        self.original_conversation_count += 1
                        self.conversations.append(conversation_content)
                except json.JSONDecodeError as e:
                    error_message = f"{e}\n"
                    print(error_message)
                    error_log.write(error_message)
                    self.error_counter[e.msg] += 1

    def analyze_errors(self):
        print("\nMost common errors:")
        for error, count in self.error_counter.most_common():
            print(f"{error}: {count} occurrences")

    def print_stats(self):
        print(f"Number of conversations in original file: {self.original_conversation_count}")
        print(f"Number of conversations in new file: {self.cleaned_conversation_count}")
        self.analyze_errors()
        
    def fix_json_format(self, text):
        # Implement logic to fix common JSON format issues
        return text  # Placeholder, needs implementation

    @staticmethod
    def remove_emojis(text):
        # Your existing emoji removal code
        pass  # Placeholder

    def clean_conversations(self):
        for section in self.conversations:
            conversation = []
            matches = re.findall(r'\"role\": \"(.*?)\",\n\"content\": \"(.*?)\"', section)
            for match in matches:
                role, content = match
                role_clean = self.remove_emojis(role.strip())
                content_clean = self.remove_emojis(content.strip())
                conversation.append({"role": role_clean, "content": content_clean})
            if conversation:
                self.cleaned_conversation_count += 1
                self.conversations_cleaned.append(conversation)
    
    def save_to_jsonl(self, output_filename):
        with open(output_filename, 'a') as outfile:
            for conversation in self.conversations_cleaned:
                outfile.write(json.dumps(conversation) + '\n')

if __name__ == "__main__":
    error_log_file = '/home/ec2-user/environment/data_generation/error_file.txt'
    processor = ConversationProcessor('/home/ec2-user/environment/data_generation/character-create/data/convo_data.jsonl', error_log_file)
    processor.load_conversations()
    processor.clean_conversations()
    processor.save_to_jsonl('/home/ec2-user/environment/data_generation/cleaned_conversations_v2.jsonl')
    processor.print_stats()
