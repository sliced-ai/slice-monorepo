import requests
import random
import openai
import os
import time
def fetch_nouns():
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
    response = requests.get(url)
    if response.status_code == 200:
        words = response.text.split('\n')
        return words
    else:
        print("Failed to fetch the list of nouns.")
        return []

def generate_seed(words):
    if len(words) >= 4:
        seed = random.sample(words, 4)
        return ' '.join(seed)
    else:
        print("Not enough words to generate seed.")
        return None

def load_multiple_prompts(file_list):
    combined_prompt = ''
    for filename in file_list:
        with open(filename, 'r') as f:
            combined_prompt += f.read().strip() + '\n'
    return combined_prompt

def load_starting_prompt(filename):
    with open(filename, 'r') as f:
        return f.read().strip()

def infer_with_gpt4(prompt, temperature=0.7, max_tokens=150):
    # Note: You'll need to install OpenAI's Python package and set up API keys.
    openai.api_key = "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"
    model_engine = "gpt-4"  # Choose the GPT-4 engine you want to use

    # Using the chat-specific endpoint
    raw_response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    full_response = raw_response['choices'][0]['message']['content'].strip()
    print("RAW RESPONSE: \n\n" + full_response)
    filtered_response = '\n'.join([
        line for line in full_response.split('\n')
        if line.strip().startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-'))
    ])

    return filtered_response

def save_response_to_file(response, filename):
    with open(filename, 'a') as f:  # Use 'a' mode for appending
        f.write(response + "\n")    # Add a newline after each response


def load_individual_prompt_and_infer(filename, core_prompt, seed, temperature=0.1, max_tokens=4000):
    with open(filename, 'r') as f:
        individual_prompt = f.read().strip()
    
    final_prompt = f"{individual_prompt}\n{core_prompt}\n\nBefore doing anything, write one sentence that uses these words: {seed}"
    #print(f"Final prompt: {final_prompt}")
    
    gpt4_response = infer_with_gpt4(final_prompt, temperature, max_tokens)
    
    save_response_to_file(gpt4_response, f'data_expansion.txt')

if __name__ == "__main__":
    # Fetch nouns from the internet
    nouns = fetch_nouns()

    # Generate a seed using 4 random nouns
    seed = generate_seed(nouns)

    # Load the core prompt from a text file
    core_prompt = load_starting_prompt('/home/ec2-user/environment/data_generation/testprompting.txt')

    if seed and core_prompt:
        folder_path = "/home/ec2-user/environment/data_generation/expansion_sets/"
        file_list = os.listdir(folder_path)

        for filename in file_list:
            print("FILE:" + filename)
            filename = folder_path + filename
            load_individual_prompt_and_infer(filename, core_prompt, seed)
            time.sleep(30)