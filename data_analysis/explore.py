import json
import os
import re

def process_json_file(file_path):
    # Open the JSON file
    with open(file_path) as file:
        data = json.load(file)

    # Initialize metrics
    total_chunks = len(data)
    total_turns = 0
    total_words = 0
    unique_names = set()
    name_utterances = {}

    # Iterate over each document (chunk)
    for document in data:
        # Update total turns
        total_turns += len(document['TURNS'])

        # Iterate over each turn
        for turn in document['TURNS']:
            # Update unique names
            unique_names.update(turn['NAMES'])

            # Update name utterances
            for name in turn['NAMES']:
                if name not in name_utterances:
                    name_utterances[name] = []
                name_utterances[name].append(turn['NUMBER'])

            # Update total words
            for utterance in turn['UTTERANCES']:
                total_words += len(utterance.split())

    # Return metrics
    return total_chunks, total_turns, total_words, unique_names, name_utterances

def process_folder(folder_path):
    # Initialize summary metrics
    total_files = 0
    total_chunks = 0
    total_turns = 0
    total_words = 0
    unique_names = set()
    common_names = None
    name_utterances = {}

    # Initialize campaign-specific metrics
    campaign_metrics = {}

    # Iterate over all files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .json extension
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                chunks, turns, words, names, file_name_utterances = process_json_file(file_path)

                # Extract the campaign number from the file name
                campaign_number = re.findall(r'C\d+', file)[0]

                # Update summary metrics
                total_files += 1
                total_chunks += chunks
                total_turns += turns
                total_words += words
                unique_names.update(names)

                # Update common names
                if common_names is None:
                    common_names = set(names)
                else:
                    common_names &= names

                # Update name utterances
                for name, utterances in file_name_utterances.items():
                    if name not in name_utterances:
                        name_utterances[name] = []
                    name_utterances[name].extend(utterances)

                # Update campaign-specific metrics
                if campaign_number not in campaign_metrics:
                    campaign_metrics[campaign_number] = {
                        'total_files': 0,
                        'total_chunks': 0,
                        'total_turns': 0,
                        'total_words': 0,
                        'unique_names': set(),
                        'common_names': None,
                        'name_utterances': {}
                    }
                campaign_metrics[campaign_number]['total_files'] += 1
                campaign_metrics[campaign_number]['total_chunks'] += chunks
                campaign_metrics[campaign_number]['total_turns'] += turns
                campaign_metrics[campaign_number]['total_words'] += words
                campaign_metrics[campaign_number]['unique_names'].update(names)

                if campaign_metrics[campaign_number]['common_names'] is None:
                    campaign_metrics[campaign_number]['common_names'] = set(names)
                else:
                    campaign_metrics[campaign_number]['common_names'] &= names

                for name, utterances in file_name_utterances.items():
                    if name not in campaign_metrics[campaign_number]['name_utterances']:
                        campaign_metrics[campaign_number]['name_utterances'][name] = []
                    campaign_metrics[campaign_number]['name_utterances'][name].extend(utterances)

    # Calculate averages
    avg_chunks_per_file = total_chunks / total_files
    avg_turns_per_file = total_turns / total_files
    avg_words_per_utterance = total_words / total_turns

    # Calculate average steps between utterances for common names
    avg_steps_between_utterances = {}
    for name in common_names:
        utterances = name_utterances[name]
        steps = [utterances[i] - utterances[i-1] for i in range(1, len(utterances))]
        avg_steps = sum(steps) / len(steps)
        avg_steps_between_utterances[name] = avg_steps

    # Calculate campaign-specific averages
    for campaign_number, metrics in campaign_metrics.items():
        metrics['avg_chunks_per_file'] = metrics['total_chunks'] / metrics['total_files']
        metrics['avg_turns_per_file'] = metrics['total_turns'] / metrics['total_files']
        metrics['avg_words_per_utterance'] = metrics['total_words'] / metrics['total_turns']

        metrics['avg_steps_between_utterances'] = {}
        for name in metrics['common_names']:
            utterances = metrics['name_utterances'][name]
            steps = [utterances[i] - utterances[i-1] for i in range(1, len(utterances))]
            avg_steps = sum(steps) / len(steps)
            metrics['avg_steps_between_utterances'][name] = avg_steps

    # Print summary metrics
    print("Summary Metrics:")
    print(f"Total Files: {total_files}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Total Turns: {total_turns}")
    print(f"Total Words: {total_words}")
    print(f"Unique Names: {', '.join(unique_names)}")
    print(f"Number of Unique Names: {len(unique_names)}")
    print(f"Average Chunks per File: {avg_chunks_per_file:.2f}")
    print(f"Average Turns per File: {avg_turns_per_file:.2f}")
    print(f"Average Words per Utterance: {avg_words_per_utterance:.2f}")
    print(f"Common Names: {', '.join(common_names)}")
    print("Average Steps Between Utterances for Common Names:")
    for name, avg_steps in avg_steps_between_utterances.items():
        print(f"  {name}: {avg_steps:.2f}")

    # Print campaign-specific metrics
    for campaign_number, metrics in campaign_metrics.items():
        print(f"\nCampaign {campaign_number} Metrics:")
        print(f"Total Files: {metrics['total_files']}")
        print(f"Total Chunks: {metrics['total_chunks']}")
        print(f"Total Turns: {metrics['total_turns']}")
        print(f"Total Words: {metrics['total_words']}")
        print(f"Unique Names: {', '.join(metrics['unique_names'])}")
        print(f"Number of Unique Names: {len(metrics['unique_names'])}")
        print(f"Average Chunks per File: {metrics['avg_chunks_per_file']:.2f}")
        print(f"Average Turns per File: {metrics['avg_turns_per_file']:.2f}")
        print(f"Average Words per Utterance: {metrics['avg_words_per_utterance']:.2f}")
        print(f"Common Names: {', '.join(metrics['common_names'])}")
        print("Average Steps Between Utterances for Common Names:")
        for name, avg_steps in metrics['avg_steps_between_utterances'].items():
            print(f"  {name}: {avg_steps:.2f}")

# Specify the folder path
folder_path = '/workspace/aligneddata'

# Process all JSON files in the folder and its subfolders
process_folder(folder_path)