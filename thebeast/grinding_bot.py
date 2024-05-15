import subprocess
import os
import openai
import anthropic
import google.generativeai as genai

# Input variables
number_of_tries = 5
models = {
    "GPT4": "gpt-4",  # Updated to the correct GPT-4 model name
    "claude": "claude-3-opus-20240229",
    "gemini": "models/gemini-pro"  # Example Gemini model name
}
order_of_calls = ["claude", "claude", "GPT4", "gemini"]  # List defining the order of model calls
base_prompt = "Solve the following coding problem:\n"
code_file_paths = ["example.py"]  # List of code file paths to be used

# API keys setup
openai.api_key = "your_openai_api_key_here"
anthropic_client = anthropic.Anthropic(api_key="your_anthropic_api_key_here")
genai.configure(api_key="your_gemini_api_key_here")

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def execute_code(file_path):
    result = subprocess.run(['python', file_path], capture_output=True, text=True)
    return result.stdout, result.stderr

def call_openai(model, prompt):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def call_claude(prompt):
    response = anthropic_client.completions.create(
        model=models["claude"],
        max_tokens_to_sample=1024,
        temperature=0.5,
        prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
    )
    return response.completion

def call_gemini(prompt):
    response = genai.generate_text(
        model=models["gemini"],
        prompt=prompt,
        temperature=0.5,
        max_output_tokens=1024
    )
    if response.result:
        return response.result
    else:
        raise ValueError("Gemini API call failed")

def call_llm(model_name, prompt):
    if model_name == "GPT4":
        return call_openai(models[model_name], prompt)
    elif model_name == "claude":
        return call_claude(prompt)
    elif model_name == "gemini":
        return call_gemini(prompt)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def format_prompt(base_prompt, code, error):
    return f"{base_prompt}\nCode:\n{code}\n\nError:\n{error}"

def main():
    call_index = 0
    for attempt in range(number_of_tries):
        for file_path in code_file_paths:
            current_code = read_file(file_path)
            output, error = execute_code(file_path)

            if not error:
                print(f"Code executed successfully on attempt {attempt + 1}.\nOutput:\n{output}")
                return

            model_name = order_of_calls[call_index % len(order_of_calls)]
            prompt = format_prompt(base_prompt, current_code, error)
            try:
                updated_code = call_llm(model_name, prompt)
            except Exception as e:
                print(f"Error calling model {model_name}: {e}")
                continue
            
            write_file(file_path, updated_code)
            print(f"Attempt {attempt + 1}, using model {model_name}, resulted in an error.\nError:\n{error}")
            call_index += 1

    print("Max attempts reached. The code did not execute successfully.")

if __name__ == "__main__":
    main()
