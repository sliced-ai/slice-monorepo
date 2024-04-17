def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        for j in range(batch_size):
            prompt = f"{base_prompt} {i}"
            dataset.append(prompt)
    return dataset

def save_dataset_to_file(dataset, file_path):
    with open(file_path, 'w') as file:
        for item in dataset:
            file.write(item + '\n')

# User input
base_prompt = ""
max_number = 100
batch_size = 4

# Generate the dataset
dataset = generate_dataset(base_prompt, max_number, batch_size)

# Save the dataset to a file
output_file = "output_dataset.txt"
save_dataset_to_file(dataset, output_file)

print(f"Dataset generated and saved to {output_file}.")