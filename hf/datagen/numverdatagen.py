import json

def generate_dataset(base_prompt, max_number, batch_size):
  dataset = []
  for i in range(max_number + 1):
    prompt_list = []  # Create a list to hold multiple prompts in the batch
    completion_list = []  # Create a list to hold multiple completions in the batch
    for j in range(batch_size):
        prompt = f"{base_prompt}" 
        completion = f"My number prediction is: {str(i)}"
        prompt_list.append(prompt)
        completion_list.append(completion)

        #print(f"i: {i}, prompt: {prompt}, completion: {completion}") 
        #print(f"Type of prompt: {type(prompt)}, type of completion: {type(completion)}")
        
        dataset.append({"prompt": prompt_list, "completion": completion_list}) 

  return dataset

def save_dataset_to_file(dataset, file_path):
    with open(file_path, 'w') as file:
        json.dump(dataset, file, indent=2)

# User input
base_prompt = "You are training in order to guess the next number as we count. I have been training you on an increasing number count. Please remember previous inputs to training and guess the next number in the sequence. What is your next number? Only respond with a single number. Assume you have started at zero."
max_number = 1000
batch_size = 4

# Generate the dataset
dataset = generate_dataset(base_prompt, max_number, batch_size)

# Save the dataset to a file
output_file = "output_dataset.json"
save_dataset_to_file(dataset, output_file)

print(f"Dataset generated and saved to {output_file}.")
