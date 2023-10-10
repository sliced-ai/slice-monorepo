import json
import re

class ConversationProcessor:
    
    def __init__(self, filename):
        self.filename = filename
        self.conversations = []
        self.conversations_cleaned = []
        
    def load_conversations(self):
        with open(self.filename, 'r') as file:
            sections = file.read().split('-----')
            for section in sections:
                try:
                    data = json.loads(section)
                    conversation_content = data.get("content", None)
                    if conversation_content:
                        self.conversations.append(conversation_content)
                except json.JSONDecodeError:
                    pass
    
    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
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
                self.conversations_cleaned.append(conversation)
    
    def save_to_jsonl(self, output_filename):
        with open(output_filename, 'a') as outfile:
            for conversation in self.conversations_cleaned:
                outfile.write(json.dumps(conversation) + '\n')

if __name__ == "__main__":
    processor = ConversationProcessor('/home/ec2-user/environment/data_generation/convo_data.jsonl')
    processor.load_conversations()
    processor.clean_conversations()
    processor.save_to_jsonl('/home/ec2-user/environment/data_generation/cleaned_conversations.jsonl')
