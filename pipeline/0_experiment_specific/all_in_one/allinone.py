import json
import requests
import os
import random
import csv
SERVICE_URL = "http://localhost:8080"

def start_service(model_name):
    data = {'model_name': model_name}
    response = requests.post(f"{SERVICE_URL}/start", json=data)
    print("Start Service:", response.text)

def generate_text(prompt, lora_params_path=None):
    data = {'prompt': prompt, 'lora_params_path': lora_params_path}
    response = requests.post(f"{SERVICE_URL}/generate", json=data)
    return response.json()

def stop_service():
    response = requests.post(f"{SERVICE_URL}/stop")
    print("Stop Service:", response.text)

def recursive_tune(dataset_path, experiment_name, iterations, lora_params_path=None):
    data = {
        'dataset_path': dataset_path,
        'experiment_name': experiment_name,
        'iterations': iterations,
        'lora_params_path': lora_params_path,
    }
    response = requests.post(f"{SERVICE_URL}/recursive_tune", json=data)
    return response.json()

def create_dataset_from_conversation(generated_text, dataset_path, min_char_length=10):
    conversations = generated_text.split('\n\n')
    valid = False
    with open(dataset_path, 'a') as f:
        for seq_id, conversation in enumerate(conversations, start=1):
            parts = conversation.split('\n')
            if len(parts) < 2:
                continue
            input_part, output_part = parts[0], parts[1]

            # Ensure 'input:' prefix is handled uniformly and remove additional ":" from output
            input_part = input_part.replace("input:", "").strip().replace(":", "")
            output_part = output_part.replace("output:", "").strip().replace(":", "")

            if len(input_part) >= min_char_length and len(output_part) >= min_char_length:
                valid = True
                # Adding "input:" and "output:" prefixes back
                formatted_input = f"input: {input_part}"
                formatted_output = f"output: {output_part}"
                data = {
                    "input": formatted_input, 
                    "output": formatted_output, 
                    "seq": seq_id
                }
                f.write(json.dumps(data) + '\n')
    return valid


def refine_generated_text(generated_text, original_prompt):
    refined_text = generated_text.replace(original_prompt, "").strip()
    lines = refined_text.split('\n')
    final_pair = lines[-2:] if len(lines) >= 2 else ""
    return '\n'.join(final_pair)

topics = [
    "bardic lore", "dwarven ale", "ancient dungeons", "mythical tales", "medieval ballads",
    "lost artifacts", "mystical enchantments", "tavern stories", "lute melodies", "heroic sagas",
    "forgotten lands", "dungeon delving", "magical instruments", "ancient runes", "legendary creatures",
    "treasure hunts", "pirate legends", "sailing adventures", "crafting songs", "romantic epics",
    "bardic competitions", "cultural festivals", "historical reenactments", "medieval fairs", "oral storytelling",
    "ancient poetry", "love tragedies", "adventurer's gear", "cloak and dagger tales", "spellcasting",
    "potion brewing", "necromancy secrets", "alchemy mysteries", "divination practices", "wizardry and sorcery",
    "folk dances", "drinking games", "runic magic", "monster lore", "dragon tales",
    "sorcerer's quests", "enchanted forests", "mythical islands", "legendary swords", "ancient prophecies"
    "celestial observations", "exotic spices", "ancient scripts", "folklore mysteries", "legendary armors",
    "magical potions", "sacred rituals", "ghost stories", "seafaring tales", "arcane libraries",
    "bardic magic", "enchanted jewelry", "mythical maps", "ancient deities", "ritualistic dances",
    "warrior poets", "mystic runes", "heroic deeds", "epic quests", "legendary battles",
    "fabled kingdoms", "mystical realms", "ancient scrolls", "sorcery and spells", "enchanted woods",
    "forgotten spells", "legendary heroes", "enchanted weapons", "mystical beasts", "hidden treasures",
    "ancient wisdom", "secret societies", "mysterious artifacts", "sacred groves", "wizarding duels",
    "mythic tales", "ancient sagas", "cursed objects", "enchanted realms", "adventurers' guilds"
]

def insert_random_topic_into_prompt(base_prompt):
    # Select a random topic from the list
    random_topic = random.choice(topics)
    # Insert the random topic into the prompt
    return base_prompt.replace("{topic}", random_topic)
    
def recursive_generation_and_training(iterations, initial_prompt, num_conversations=8):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    experiment_name = "dynamic_experiment_6"
    lora_params_path = None  # Start without any LoRA parameters
    
    for i in range(iterations):
        dataset_path = f"/home/ec2-user/environment/pipeline/all_in_one/data/{experiment_name}/iteration_{i}/dataset.jsonl"
        failure_counter_path = f"/home/ec2-user/environment/pipeline/all_in_one/data/{experiment_name}/failure_counter.csv"
        dataset_folder = f"/home/ec2-user/environment/pipeline/all_in_one/data/{experiment_name}/iteration_{i}/"
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Ensure the service is started outside the while loop to avoid unnecessary restarts
        start_service(model_name)
        
        conversation_parts = 0
        failure_counter = 0
        while conversation_parts < num_conversations:
            input_prompt = insert_random_topic_into_prompt(initial_prompt)
            generated_response = generate_text(input_prompt, lora_params_path)
            generated_text = generated_response.get("generated_text", "")
            refined_text = refine_generated_text(generated_text, input_prompt)
            
            if create_dataset_from_conversation(refined_text, dataset_path):
                conversation_parts += 1
            else:
                print("Regenerating text due to validation failure.")
                failure_counter += 1

        with open(failure_counter_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Check if the file is empty to write headers
            if os.stat(failure_counter_path).st_size == 0:
                csvwriter.writerow(['Iteration', 'Failures'])
            csvwriter.writerow([i, failure_counter])             
        stop_service()
        start_service(model_name)
        
        # Assuming recursive_tune now correctly takes the dataset folder path
        response = recursive_tune(dataset_folder, experiment_name, 1, lora_params_path)
        lora_params_path = response.get("final_lora_params_path")
        stop_service()
    
    stop_service()

Input_prompt= """Given this description, create a synthetic conversation between this character, Geth Stormsong, and another character. Start with the other character speaking to Geth.

Description of Geth Stormsong: An old human bard with close cropped hair and beard, named Geth Stormsong. He is of normal height but has a bit of a beer gut from drinking with the dwarves in the north where he often quests. He wears coin on clothes but for a deep purple cloak he acquired in a dungeon. Geth also wears two gold rings on the ring finger of his left hand in memory of a wife he had in his youth. She passed in childbirth and Geth never sought anyone else. He wears a coin from an unidentifiable land on a cord around his neck from his homeland. People always ask where it's from, but he just smiles and shakes his head. He keeps the old lute slung across his back.

Topic: life
Example:
input: Hi Geth tell me about your home?
output: Arg I'm doing swell lad, let's get a drink!

Topic: {topic}
Create:
input: 
output: 
"""

#recursive_generation_and_training(100, Input_prompt)
# Example call
recursive_generation_and_training(1000, Input_prompt, num_conversations=50)
