import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
from textblob import TextBlob
import nltk

# Normalizing Function
def normalize(value, max_value):
    return value / max_value if max_value != 0 else 0

# N-gram Diversity
def ngram_diversity(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text])
    return len(vectorizer.get_feature_names())

# Type-Token Ratio
def ttr(text):
    tokens = nltk.word_tokenize(text.lower())
    types = set(tokens)
    return len(types) / len(tokens) if len(tokens) > 0 else 0

# Sentiment Diversity Normalized
def sentiment_diversity_normalized(sentences, max_sentiment_count):
    sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
    sentiment_counts = Counter(['positive' if s > 0 else 'negative' if s < 0 else 'neutral' for s in sentiments])
    normalized_sentiments = {k: normalize(v, max_sentiment_count) for k, v in sentiment_counts.items()}
    return normalized_sentiments

# Constants for normalization
MAX_UNIGRAM = 1000
MAX_BIGRAM = 1000
MAX_TRIGRAM = 1000
MAX_SENTIMENT_COUNT = 100

# Read texts
with open("/home/ec2-user/environment/data_generation/character-create/generated_texts.txt", "r") as f:
    content = f.read()
    texts = content.split("\nFull Name:")

# Initialize ranks
ranks = defaultdict(int)

# Radar chart preparation
metrics = [
    'Normalized Unigram Diversity', 
    'Normalized Bigram Diversity', 
    'Normalized Trigram Diversity', 
    'TTR', 
    'Normalized Positive Sentiment', 
    'Normalized Neutral Sentiment', 
    'Normalized Negative Sentiment'
]
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
fig.subplots_adjust(right=0.5)

# Processing and ranking texts
for text in texts:
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    unigram_score = normalize(ngram_diversity(text, 1), MAX_UNIGRAM)
    bigram_score = normalize(ngram_diversity(text, 2), MAX_BIGRAM)
    trigram_score = normalize(ngram_diversity(text, 3), MAX_TRIGRAM)
    ttr_score = ttr(text)
    normalized_sentiments = sentiment_diversity_normalized(sentences, MAX_SENTIMENT_COUNT)

    scores = [
        unigram_score,
        bigram_score,
        trigram_score,
        ttr_score,
        normalized_sentiments.get('positive', 0),
        normalized_sentiments.get('neutral', 0),
        normalized_sentiments.get('negative', 0)
    ]

    # Update the ranks
    ranks[text] += sum(scores)

    # Radar chart plotting
    ax.plot(angles, scores, linewidth=2, label=text[:30])  # Truncate label to first 30 chars
    ax.fill(angles, scores, 'none')

# Highest Overall Diversity Text
best_text = max(ranks, key=ranks.get)
print(f"The text with the highest overall diversity is: {best_text[:30]}")  # Truncate to first 30 characters


# Radar chart labels and legend
ax.set_xticks(angles)
ax.set_xticklabels(metrics)
plt.legend(loc='upper left', bbox_to_anchor=(1.3, 0.8))

# Title and saving
plt.title("Text Analysis Metrics Across Multiple Texts")
plt.savefig("multi_text_analysis_metrics.png")
