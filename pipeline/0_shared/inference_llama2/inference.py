import requests

SERVICE_URL = "http://localhost:5000"

def start_service(model_name):
    data = {
        'model_name': model_name,
    }
    response = requests.post(f"{SERVICE_URL}/start", json=data)
    print(response.json())

def generate_text(prompt):
    data = {'prompt': prompt}
    response = requests.post(f"{SERVICE_URL}/generate", json=data)
    print(response.json())

def stop_service():
    response = requests.post(f"{SERVICE_URL}/stop")
    print(response.json())

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Update this path
    
    start_service(model_name)
    
    prompts = [
        #"Generate a conversation based on this template: While looking for [Random Noun 1], [Character 1] asked [Character 2] if they had seen it, accidentally uncovering [Character 2]'s previously unknown interest in [Random Noun 2].",
        #"Generate a conversation based on this template: Curious about [Random Noun 1]'s origin, [Character 1] inquired [Character 2] about where it came from, leading to a fascinating tale about [Character 2]'s adventure involving [Random Noun 2].",
        #"Generate a conversation based on this template: During a conversation about hobbies, [Character 1] asked [Character 2] how they got started with [Random Noun 1], which led to [Character 2] sharing a heartfelt story about [Random Noun 2].",
        #"Generate a conversation based on this template: [Character 1] noticed [Character 2] seemed particularly focused on [Random Noun 1] and asked why it was so significant, revealing a deep connection between [Character 2] and [Random Noun 2]",
        "Generate a character description, This character is 5 years old and just getting memories, including:\n- Full Name\n- Nickname (if any)\n- Age\n- Gender\n- Ethnicity\n- Nationality\n- Height\n- Weight\n- Hair Color\n- Eye Color\n- Scars or Tattoos (if applicable)\n- Clothing Style (suitable for the character's age and preferences)\n- Hobbies (age-appropriate activities)\n- Favorite Food\n- Language Proficiency\n- Family (parents, siblings, other relatives)\n- Friends (if any, appropriate to the character's social circle)\n- Education (level appropriate for age)\n- Occupational History (if applicable)\n- Past Year Overview: Key experiences and developments over the past year.\n- Current Feelings and State: General emotional and mental state.\n- Motivations\n- Fears\n- Secrets (if any)\n\nKeep the character realistic for their age and background. The output format is a list of bullet points. Incorporate {random_seed} within the character's description."
    ]
    for prompt in prompts:
        generate_text(prompt)
    
    #stop_service()
