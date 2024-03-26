import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from datasketch import MinHash
import spacy
import seaborn as sns
import pandas as pd
from matplotlib.colors import to_rgba

# Ensure you have downloaded the necessary NLTK data and spaCy model
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

class TextQualityAnalyzer:

    def __init__(self, max_unigram, max_bigram, max_trigram, max_quadgram):
        self.MAX_UNIGRAM = max_unigram
        self.MAX_BIGRAM = max_bigram
        self.MAX_TRIGRAM = max_trigram
        self.MAX_QUADGRAM = max_quadgram
        self.previous_texts_minhash = []

    @staticmethod
    def normalize(value, max_value):
        return value / max_value if max_value != 0 else 0

    @staticmethod
    def ngram_diversity(text, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word', token_pattern=r'\b\w+\b')
        ngrams = vectorizer.fit_transform([text])
        return len(vectorizer.get_feature_names())

    @staticmethod
    def ttr(text):
        tokens = word_tokenize(text.lower())
        types = set(tokens)
        return len(types) / len(tokens) if len(tokens) > 0 else 0

    def analyze_texts(self, input_texts):
        combined_text = ' '.join(input_texts)
        scores = [
            self.normalize(self.ngram_diversity(combined_text, 1), self.MAX_UNIGRAM),
            self.normalize(self.ngram_diversity(combined_text, 2), self.MAX_BIGRAM),
            self.normalize(self.ngram_diversity(combined_text, 3), self.MAX_TRIGRAM),
            self.normalize(self.ngram_diversity(combined_text, 4), self.MAX_QUADGRAM),
            self.ttr(combined_text),
            self.calculate_novelty(combined_text),
        ]

        metrics = [
            'Unigram Diversity',
            'Bigram Diversity',
            'Trigram Diversity',
            'Quadgram Diversity',
            'TTR',
            'Novelty',
        ]

        analysis_results = dict(zip(metrics, scores))
        return analysis_results

    def calculate_novelty(self, text):
        # Create a MinHash for the current text
        current_minhash = MinHash()
        for token in nlp(text):
            current_minhash.update(token.text.encode('utf8'))
        # Calculate similarity with previous texts
        novelty_scores = [current_minhash.jaccard(pm) for pm in self.previous_texts_minhash] or [1]
        # Save the MinHash of the current text for future comparisons
        self.previous_texts_minhash.append(current_minhash)
        # Novelty score is the complement of average similarity
        return 1 - np.mean(novelty_scores)

def load_conversations_from_file(file_path):
    conversations = []
    with open(file_path, 'r') as file:
        for line in file:
            conversation = json.loads(line)
            conversations.append(conversation["input"] + " " + conversation["output"])
    return conversations

def visualize_analysis_over_iterations_combined(base_dir, experiment_name, output_png):
    analyzer = TextQualityAnalyzer(max_unigram=1000, max_bigram=1000, max_trigram=1000, max_quadgram=1000)
    metrics_distribution = defaultdict(list)
    all_conversations = set()

    iterations = sum(os.path.isdir(os.path.join(base_dir, experiment_name, f'iteration_{i}')) for i in range(100))

    for iteration_number in range(iterations):
        dataset_path = f"{base_dir}/{experiment_name}/iteration_{iteration_number}/dataset.jsonl"
        if not os.path.exists(dataset_path):
            continue
        conversations = load_conversations_from_file(dataset_path)
        
        for convo in conversations:
            is_repeat = convo in all_conversations
            all_conversations.add(convo)
            individual_analysis = analyzer.analyze_texts([convo])  # Analyze individually
            for key, value in individual_analysis.items():
                metrics_distribution[key].append((iteration_number, value))
            metrics_distribution["Repeats"].append((iteration_number, int(is_repeat)))  # Marking repeat as binary

    num_metrics = len(metrics_distribution)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics), dpi=200)

    colors = sns.color_palette("Spectral", num_metrics)
    
    for (metric, data), ax, color in zip(metrics_distribution.items(), axs, colors):
        df = pd.DataFrame(data, columns=['Iteration', metric])
        sns.violinplot(x="Iteration", y=metric, data=df, ax=ax, palette=[color])
        ax.set_title(metric)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)  # Save as a high-definition image
    plt.show()

# Example usage
base_dir = "/home/ec2-user/environment/pipeline/all_in_one/data"
experiment_name = "dynamic_experiment_4"
output_png = f"/home/ec2-user/environment/pipeline/all_in_one/data/{experiment_name}/analysis_over_iterations"
visualize_analysis_over_iterations_combined(base_dir, experiment_name, output_png)