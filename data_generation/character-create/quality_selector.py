import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
import json
from textblob import TextBlob
import nltk


class TextQualityAnalyzer:

    def __init__(self, max_unigram, max_bigram, max_trigram, max_sentiment_count):
        self.MAX_UNIGRAM = max_unigram
        self.MAX_BIGRAM = max_bigram
        self.MAX_TRIGRAM = max_trigram
        self.MAX_SENTIMENT_COUNT = max_sentiment_count

    @staticmethod
    def normalize(value, max_value):
        return value / max_value if max_value != 0 else 0

    @staticmethod
    def ngram_diversity(text, n):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngrams = vectorizer.fit_transform([text])
        return len(vectorizer.get_feature_names())

    @staticmethod
    def ttr(text):
        tokens = nltk.word_tokenize(text.lower())
        types = set(tokens)
        return len(types) / len(tokens) if len(tokens) > 0 else 0

    def sentiment_diversity_normalized(self, sentences):
        sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
        sentiment_counts = Counter(['positive' if s > 0 else 'negative' if s < 0 else 'neutral' for s in sentiments])
        normalized_sentiments = {k: self.normalize(v, self.MAX_SENTIMENT_COUNT) for k, v in sentiment_counts.items()}
        return normalized_sentiments

    def analyze_texts(self, input_file, output_json, output_png):
        with open(input_file, 'r') as f:
            content = f.read()
            texts_with_names = content.split("\nFull Name:")
            
        ranks = defaultdict(int)
        
        data_for_json = {}
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
        fig.subplots_adjust(right=0.5)
    
        for full_text in texts_with_names[1:]:  # Skip the first empty string
            text = "Full Name:" + full_text  # Add back the full name to each text
            sentences = nltk.sent_tokenize(full_text)  # Remove full name for analysis
            sentiment_scores = self.sentiment_diversity_normalized(sentences)
            
            scores = [
                self.normalize(self.ngram_diversity(full_text, 1), self.MAX_UNIGRAM),
                self.normalize(self.ngram_diversity(full_text, 2), self.MAX_BIGRAM),
                self.normalize(self.ngram_diversity(full_text, 3), self.MAX_TRIGRAM),
                self.ttr(full_text),
                *sentiment_scores.values()
            ]
            
            metrics = [
                'Normalized Unigram Diversity',
                'Normalized Bigram Diversity',
                'Normalized Trigram Diversity',
                'TTR',
                *sentiment_scores.keys()
            ]
            
            ranks[text] = sum(scores)  # Using full text including name for ranking
            data_for_json[text[:30]] = dict(zip(metrics, scores))
    
            num_vars = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            ax.plot(angles, scores, linewidth=2, label=full_text[:20])
            ax.fill(angles, scores, 'none')
    
            ax.set_xticks(angles)
            ax.set_xticklabels(metrics)
    
        best_text = max(ranks, key=ranks.get)  # This will now have the full name as well
        data_for_json['best_text'] = best_text  # This will now have the full name as well
    
        with open(output_json, 'w') as f:
            json.dump(data_for_json, f)
    
        plt.legend(loc='upper left', bbox_to_anchor=(1.3, 0.8))
        plt.title("Text Analysis Metrics Across Multiple Texts")
        plt.savefig(output_png)
    
        return best_text  # This will now be the complete text including the full name

def analyze_and_select_best_text(input_file, output_json, output_png, max_unigram=1000, max_bigram=1000, max_trigram=1000, max_sentiment_count=100):
    analyzer = TextQualityAnalyzer(max_unigram, max_bigram, max_trigram, max_sentiment_count)
    return analyzer.analyze_texts(input_file, output_json, output_png)
