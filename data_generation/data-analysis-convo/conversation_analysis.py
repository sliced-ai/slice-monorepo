
import os
import json
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from collections import defaultdict

# Function to load conversations from a specific range in a JSONL file
def load_conversations(file_path, ranges):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            for start, end in ranges:
                if start <= i < end:
                    conversation = json.loads(line.strip())
                    conversations.append(conversation)
                    break
    return conversations

# Function to print basic information about the conversations
def print_basic_info(conversations):
    print(f"Total conversations loaded: {len(conversations)}")
    roles = set()
    for convo in conversations:
        for exchange in convo:
            roles.add(exchange['role'])
    print(f"Roles involved: {', '.join(roles)}")
    print("Sample conversation:")
    for exchange in conversations[0]:
        print(f"{exchange['role']}: {exchange['content']}")

# Function to create batches from data
def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]

# Function to compute embeddings for the conversations in batches
def compute_embeddings(model, tokenizer, conversations, batch_size=50):
    model = model.cuda()
    conversation_lines = [" ".join([line['content'] for line in convo]) for convo in conversations]
    embeddings = []
    for batch_lines in batch(conversation_lines, batch_size):
        inputs = tokenizer(batch_lines, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Function to analyze similarity between conversations
def analyze_similarity(conversations):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = compute_embeddings(model, tokenizer, conversations)

    similarity_scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarity_scores.append(sim)

    return similarity_scores

# Function to save histogram of similarity scores
def save_histogram(similarity_scores, file_name):
    plt.hist(similarity_scores, bins=200, color='blue', alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Number of Conversation Pairs')
    plt.title('Distribution of Conversation Similarity Scores')
    plt.savefig(file_name)
    plt.close()

def analyze_roles(file_path):
    role_mentions = {}
    role_words = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            conversation = json.loads(line.strip())
            for exchange in conversation:
                role = exchange['role']
                content = exchange['content']
                words = content.split()
                word_count = len(words)

                if role not in role_mentions:
                    role_mentions[role] = 0
                    role_words[role] = []

                role_mentions[role] += 1
                role_words[role].append(word_count)

    # Calculate statistics and create a DataFrame
    role_stats = []
    for role, counts in role_words.items():
        total_words = sum(counts)
        average_words = np.mean(counts)
        std_dev_words = np.std(counts)

        role_stats.append({
            'Role': role,
            'Mentions': role_mentions[role],
            'Total Words': total_words,
            'Average Words': average_words,
            'Std Dev Words': std_dev_words
        })

    df = pd.DataFrame(role_stats)
    df_sorted = df.sort_values(by='Total Words', ascending=False).reset_index(drop=True)
    
    # Print the DataFrame
    print(df_sorted)

    return df_sorted


def save_similarity_to_csv(similarity_scores, num_conversations, filename):
    """
    Saves the similarity scores by conversation number to a CSV file.
    
    :param similarity_scores: List of similarity scores between conversations.
    :param num_conversations: Total number of conversations analyzed.
    :param filename: Name of the CSV file to save.
    """
    # Generate conversation pairs based on the number of conversations
    conversation_pairs = list(itertools.combinations(range(num_conversations), 2))

    # Ensure the number of scores matches the number of conversation pairs
    assert len(similarity_scores) == len(conversation_pairs), "Number of similarity scores and conversation pairs do not match."
    
    # Create a DataFrame with the conversation pairs and similarity scores
    df = pd.DataFrame({
        'Conversation Pair': [f"{i}-{j}" for i, j in conversation_pairs],
        'Similarity Score': similarity_scores
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def remove_duplicates(jsonl_file_path):
    unique_conversations = set()
    cleaned_conversations = []

    # Load all conversations and add them to a set for uniqueness
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line not in unique_conversations:
                unique_conversations.add(line)
                cleaned_conversations.append(json.loads(line))

    # Save the unique conversations back to the file
    with open(jsonl_file_path, 'w', encoding='utf-8') as file:
        for conversation in cleaned_conversations:
            file.write(json.dumps(conversation) + '\n')

    print(f"Cleaned data saved to {jsonl_file_path}")

def filter_and_save_conversations(similarity_file, jsonl_file_path, output_file_path, threshold,data_dir):
    # Load the similarity scores into a DataFrame
    similarity_df = pd.read_csv(similarity_file)
    
    # Calculate median, average, and std similarity scores
    similarity_df[['Convo1', 'Convo2']] = similarity_df['Conversation Pair'].str.split('-', expand=True)
    grouped_similarity = similarity_df.groupby('Convo1')['Similarity Score']
    convo1_medians = grouped_similarity.median()
    convo1_means = grouped_similarity.mean()
    convo1_stds = grouped_similarity.std()

    # Repeat for the second conversation in each pair
    grouped_similarity = similarity_df.groupby('Convo2')['Similarity Score']
    convo2_medians = grouped_similarity.median()
    convo2_means = grouped_similarity.mean()
    convo2_stds = grouped_similarity.std()

    # Combine the stats and reset the index
    median_scores = pd.concat([convo1_medians, convo2_medians]).groupby(level=0).median()
    mean_scores = pd.concat([convo1_means, convo2_means]).groupby(level=0).mean()
    std_scores = pd.concat([convo1_stds, convo2_stds]).groupby(level=0).mean()

    # Identify conversations to remove based on the threshold
    conversations_to_remove = median_scores[median_scores > threshold].index.tolist()

    # Create a DataFrame from the JSONL file for conversation indices
    with open(jsonl_file_path, 'r', encoding='utf-8') as infile:
        conversations = [json.loads(line.strip()) for line in infile]
    convo_indices_df = pd.DataFrame({'Conversation Index': range(len(conversations))})
    
    # Prepare DataFrames for plotting
    median_scores_df = pd.DataFrame({'Median Similarity Score': median_scores}).reset_index()
    mean_scores_df = pd.DataFrame({'Mean Similarity Score': mean_scores}).reset_index()
    std_scores_df = pd.DataFrame({'STD Similarity Score': std_scores}).reset_index()
    
    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(convo_indices_df['Conversation Index'], median_scores_df['Median Similarity Score'], 
                color='blue', s=10, label='Median')
    plt.scatter(convo_indices_df['Conversation Index'], mean_scores_df['Mean Similarity Score'], 
                color='red', s=10, label='Mean')
    plt.scatter(convo_indices_df['Conversation Index'], std_scores_df['STD Similarity Score'], 
                color='green', s=10, label='STD')
    plt.title('Similarity Scores per Conversation Index')
    plt.xlabel('Conversation Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'similarity_scores_vs_index.png'))
    plt.close()

    # Filter out conversations above the similarity threshold
    filtered_conversations = [conv for i, conv in enumerate(conversations) if str(i) not in conversations_to_remove]
    
    # Save the filtered conversations to a new JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for conv in filtered_conversations:
            outfile.write(json.dumps(conv) + '\n')

    # Plotting the histogram of median similarity scores
    plt.figure(figsize=(10, 6))
    plt.hist(median_scores.values, bins=50, color='blue', alpha=0.7, label='Median Similarity Scores')
    plt.title('Histogram of Median Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'histogram_median_similarity_scores.png'))
    plt.close()
    
    print(f"Filtered conversations saved to {output_file_path}")
