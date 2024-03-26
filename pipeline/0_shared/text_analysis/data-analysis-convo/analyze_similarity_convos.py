import os
import logging
import json
import pandas as pd
from conversation_analysis import (
    remove_duplicates,
    load_conversations,
    print_basic_info,
    analyze_similarity,
    save_histogram,
    save_similarity_to_csv,
    analyze_roles,
    filter_and_save_conversations
)

from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configuration
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', '01-17-24-23yo-role-fix')
DATA_PATH = os.getenv('DATA_PATH', '/home/ec2-user/environment/data_generation/data-analysis-convo')
DATA_DIR = f'{DATA_PATH}/data/{EXPERIMENT_NAME}'
os.makedirs(DATA_DIR, exist_ok=True)
CHARACTER_NAMES = ["Maxwell James Thompson","Max"]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Paths
    file_path = f'{DATA_PATH}/filtered_conversations_modified.jsonl'
    similarity_file = os.path.join(DATA_DIR, 'similarity_scores.csv')
    output_file_path = os.path.join(DATA_DIR, 'filtered_conversations.jsonl')
    histogram_path = os.path.join(DATA_DIR, 'similarity_scores_histogram.png')
    roles_output_path = os.path.join(DATA_DIR, 'role_analysis.tsv')

    ranges = [(0, 100000)]
    threshold = 0.7  # Could also be configured through an environment variable

    # Process the conversations
    remove_duplicates(file_path)
    conversations = load_conversations(file_path, ranges)
    print_basic_info(conversations)

    # Similarity analysis
    #similarity_scores = analyze_similarity(conversations)
    #save_histogram(similarity_scores, histogram_path)
    #save_similarity_to_csv(similarity_scores, len(conversations), similarity_file)

    # Role analysis + role fix
    roles_df = analyze_roles(file_path,CHARACTER_NAMES)
    roles_df.to_csv(roles_output_path, sep='\t', index=False)

    # Filtering conversations
    #filter_and_save_conversations(similarity_file, file_path, output_file_path, threshold,DATA_DIR)

if __name__ == '__main__':
    main()
