# Script to categorize contents of a text file based on specific patterns

import re

def process_and_save(characters, items):
    # This is a placeholder for where you would implement your processing logic.
    # For demonstration purposes, let's assume we're just combining the lists into one string.
    combined_data = "Characters:\n" + "\n".join(characters) + "\n\nItems:\n" + "\n".join(items)
    
    # Now, we save the combined data to a text file.
    output_file_path = 'processed_data_output.txt'  # Define your output file name and path here
    
    with open(output_file_path, 'a') as output_file:
        output_file.write(combined_data)
    
    print(f"Data processed and saved to {output_file_path}")

def parse_and_categorize(file_path):

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            items = []  # List to hold items within brackets
            characters = []  # List to hold '[Character]' occurrences
            # Find all occurrences of bracketed items
            bracketed_items = re.findall(r'\[([^\]]+)\]', line)

            for item in bracketed_items:
                # If the item is 'Character', add to the characters list
                if re.match(r'Character.*', item, re.IGNORECASE):
                    characters.append(item)
                else:
                    # Otherwise, add to the general items list
                    items.append(item)
            process_and_save(characters, items)

# Example usage
if __name__ == "__main__":
    file_path = '/home/ec2-user/environment/pipeline/1_details_generation/detail_templates.txt'  # Replace with your actual file path
    parse_and_categorize(file_path)