# Import necessary libraries
import json
from generate import TextGenerator  # Replace with the actual name of your file containing TextGenerator

class ConversationStarter:
    def __init__(self, text_generator, meta_file_path, conversations_file_path):
        self.text_generator = text_generator
        self.meta_file_path = meta_file_path
        self.conversations_file_path = conversations_file_path

    def get_main_character_name(self):
        with open(self.meta_file_path, 'r') as f:
            main_character_name = f.readline().strip()
        return main_character_name

    def modify_conversation_start(self):
        main_character_name = self.get_main_character_name()
        
        new_file_path = self.conversations_file_path.replace('.jsonl', '_modified.jsonl')

        with open(new_file_path, 'w') as output_file:
            with open(self.conversations_file_path, 'r') as input_file:
                for line in input_file:
                    conversation = json.loads(line)
                    if conversation[0]['role'] == main_character_name:
                        new_start = self.generate_new_start(conversation)
                        conversation.insert(0, {"role": "Input", "content": new_start})
                    
                    json.dump(conversation, output_file)
                    output_file.write('\n')


    def generate_new_start(self, conversation):
        # Format the conversation into a string
        conversation_str = ' '.join([f"{message['role']}: {message['content']}" for message in conversation])
    
        # In-context learning prompt with examples
        example_prompt = (
            "Provide a New Start sentence for example 3 based on the examples below:\n\n"
            "Example1: \"*sighs* I don't get why I'm always left out of things. My classmates always invite each other to play games or go on field trips, but they never invite me. It feels like they don't want me around.\"\n"
            "New Start1: \"Hey, I've noticed you've been looking a bit down lately. What's been going on with you?\"\n\n"
            "Example2: \"Hi there! *excitedly* I'm so glad you're here! *bounces up and down* My name is Max, and I'm 7 years old. *giggles* What's your name?\"\n"
            "New Start2: \"Hello! You seem really happy and full of energy today. I'm curious, what's making you so excited?\"\n\n"
            f"Example3: \"{conversation_str}\"\n"
            "New Start3: "
        )
    
        response = self.text_generator.generate_text(example_prompt)
        new_start = response.split('New Start3: ')[-1].strip()
        return new_start


# Example usage
if __name__ == "__main__":
    # Initialize the TextGenerator
    text_generator = TextGenerator(model_engine='your_model_engine', api_key='your_api_key',
                                   use_llama=True, ckpt_dir="./model/llama-2-7b-chat", tokenizer_path="./model/tokenizer.model")

    # Create an instance of ConversationStarter
    conversation_starter = ConversationStarter(text_generator, './data/meta.txt', './data/filtered_conversations.jsonl')

    # Modify the conversation starts
    conversation_starter.modify_conversation_start()
