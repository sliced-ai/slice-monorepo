import re
import textstat
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from textblob import TextBlob
import nltk
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')


class TextQualityAnalyzer:

    def __init__(self, max_unigram, max_bigram, max_trigram, max_sentiment_count):
        assert isinstance(max_unigram, (int, float)), "max_unigram should be int or float"
        assert isinstance(max_bigram, (int, float)), "max_bigram should be int or float"
        assert isinstance(max_trigram, (int, float)), "max_trigram should be int or float"
        assert isinstance(max_sentiment_count, (int, float)), "max_sentiment_count should be int or float"
        
        self.MAX_UNIGRAM = max_unigram
        self.MAX_BIGRAM = max_bigram
        self.MAX_TRIGRAM = max_trigram
        self.MAX_SENTIMENT_COUNT = max_sentiment_count

    @staticmethod
    def lexical_density(text):
        tokens = nltk.word_tokenize(text.lower())
        total_tokens = len(tokens)
        total_unique_tokens = len(set(tokens))
        return total_unique_tokens / total_tokens if total_tokens > 0 else 0

    @staticmethod
    def hapax_legomena(text):
        tokens = nltk.word_tokenize(text.lower())
        frequency = Counter(tokens)
        hapax = sum(1 for count in frequency.values() if count == 1)
        return hapax / len(tokens) if len(tokens) > 0 else 0

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

    @staticmethod
    def average_sentence_length(text):
        sentences = nltk.sent_tokenize(text)
        total_words = sum([len(nltk.word_tokenize(sentence)) for sentence in sentences])
        return total_words / len(sentences) if len(sentences) > 0 else 0

    @staticmethod
    def readability(text):
        return textstat.flesch_reading_ease(text)

    @staticmethod
    def syntactic_complexity(text):
        sentences = nltk.sent_tokenize(text)
        total_clauses = sum([len(list(nlp(sentence).sents)) for sentence in sentences])
        total_dependent_clauses = sum([len([token for token in nlp(sentence) if "relcl" in token.dep_ or "acl" in token.dep_]) for sentence in sentences])
        return total_dependent_clauses / total_clauses if total_clauses > 0 else 0

    @staticmethod
    def local_coherence(text):
        sentences = nltk.sent_tokenize(text)
        # Using Spacy's word vectors to get embeddings for each sentence.
        sentence_embeddings = [nlp(sentence).vector for sentence in sentences]
        
        coherence_scores = []
        for i in range(1, len(sentence_embeddings)):
            coherence_scores.append(cosine_similarity([sentence_embeddings[i-1]], [sentence_embeddings[i]])[0][0])
        
        return np.mean(coherence_scores) if coherence_scores else 0

    @staticmethod
    def reference_resolution(text):
        doc = nlp(text)
        unresolved_references = [token.text for token in doc if token.dep_ in ['nsubj', 'dobj'] and (not token.head.text or not token.head.lemma_)]
        return 1 - (len(unresolved_references) / len(doc)) if len(doc) > 0 else 0


    @staticmethod
    def semantic_depth(text):
        doc = nlp(text)
        return len([ent for ent in doc.ents]) / len(nltk.word_tokenize(text)) if nltk.word_tokenize(text) else 0
    
    @staticmethod
    def compute_similarity_between_sentences(sentences):
        """
        Compute and return the cosine similarity between each pair of sentences.
        """
        sentence_embeddings = [nlp(sent).vector for sent in sentences]
        similarity_matrix = cosine_similarity(sentence_embeddings)
        similarity_dict = {}

        for idx1, sent1 in enumerate(sentences):
            for idx2, sent2 in enumerate(sentences):
                if idx1 != idx2:  # We skip similarity of a sentence with itself (always 1)
                    key = f"Sent({idx1+1}) to Sent({idx2+1})"
                    similarity_dict[key] = similarity_matrix[idx1][idx2]

        return similarity_dict
    
    
    def sentiment_diversity_normalized(self, sentences):
        sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
        sentiment_counts = Counter(['positive' if s > 0 else 'negative' if s < 0 else 'neutral' for s in sentiments])
        
        # Ensure all sentiments are present in the output
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
                
        normalized_sentiments = {k: self.normalize(v, self.MAX_SENTIMENT_COUNT) for k, v in sentiment_counts.items()}
        return normalized_sentiments

    @staticmethod
    def is_duplicate(text1, text2, threshold=0.95):
        """
        Check if two text blobs are duplicates based on their sentence embeddings.
        Compares each sentence from text1 with every sentence from text2.
        """
        sentences1 = nltk.sent_tokenize(text1)
        sentences2 = nltk.sent_tokenize(text2)

        for sent1 in sentences1:
            vector1 = nlp(sent1).vector
            for sent2 in sentences2:
                vector2 = nlp(sent2).vector
                similarity = cosine_similarity([vector1], [vector2])[0][0]
                if similarity >= threshold:
                    return True
        return False

    @staticmethod
    def preprocess(text):
        # Noise removal (URLs, special characters, etc.)
        cleaned_text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\@\w+|\#', '', cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
        # Convert to lowercase and tokenize
        cleaned_text = nltk.word_tokenize(cleaned_text.lower())
        return cleaned_text

    def compute_composite_score(self, metric_scores, metric_weights=None):
        """
        Compute a composite score for a text based on its metric scores and the given weights.
        """
        # If no weights are provided, assume equal importance for all metrics
        if not metric_weights:
            metric_weights = {metric: 1 for metric in metric_scores}

        weighted_scores = [metric_scores[metric] * metric_weights.get(metric, 1) for metric in metric_scores if metric != "Original Text"]
        return sum(weighted_scores) / sum(metric_weights.values())

    def analyze_texts(self, text_dict, metric_weights):
        analysis_results = {}
        ranking_results = {}
        processed_texts = []
        sentence_similarity_results = {}
    
        for text_id, full_text in text_dict.items():
            cleaned_text = self.preprocess(full_text)

            # Use full_text instead of cleaned_text for duplicate check
            if any(self.is_duplicate(full_text, processed_text) for processed_text in processed_texts):
                print("DUPLICATE FOUND for Text ID:", text_id)
                continue
    
            processed_texts.append(full_text)  # append full_text after checking for duplicates
            
            # Compute similarity between each pair of sentences
            sentences = nltk.sent_tokenize(full_text)
            if len(sentences) > 1:
                sentence_similarity_results[text_id] = self.compute_similarity_between_sentences(sentences)
            
            sentiment_scores = self.sentiment_diversity_normalized(nltk.sent_tokenize(full_text))
            
            scores = {
                'Original Text': cleaned_text,
                'Normalized Unigram Diversity': self.normalize(self.ngram_diversity(' '.join(cleaned_text), 1), self.MAX_UNIGRAM),
                'Normalized Bigram Diversity': self.normalize(self.ngram_diversity(' '.join(cleaned_text), 2), self.MAX_BIGRAM),
                'Normalized Trigram Diversity': self.normalize(self.ngram_diversity(' '.join(cleaned_text), 3), self.MAX_TRIGRAM),
                'TTR': self.ttr(' '.join(cleaned_text)),
                'Local Coherence': self.local_coherence(' '.join(cleaned_text)),
                'Reference Resolution': self.reference_resolution(' '.join(cleaned_text)),
                'Lexical Density': self.lexical_density(' '.join(cleaned_text)),
                'Hapax Legomena': self.hapax_legomena(' '.join(cleaned_text)),
                'Average Sentence Length': self.average_sentence_length(' '.join(cleaned_text)),
                'Readability Score': self.readability(' '.join(cleaned_text)),
                'Syntactic Complexity': self.syntactic_complexity(' '.join(cleaned_text)),
                'Semantic Depth': self.semantic_depth(' '.join(cleaned_text)),
                **sentiment_scores
                }
            analysis_results[text_id] = scores
            # Calculate composite score for the text and store it
            composite_score = self.compute_composite_score(scores, metric_weights)
            ranking_results[text_id] = composite_score
        
        ranked_texts = sorted(ranking_results.keys(), key=lambda x: ranking_results[x], reverse=True)

        return analysis_results,ranked_texts,sentence_similarity_results

def analyze_texts(text_dict, metric_weights, max_unigram=1000, max_bigram=1000, max_trigram=1000, max_sentiment_count=100):
    analyzer = TextQualityAnalyzer(max_unigram, max_bigram, max_trigram, max_sentiment_count)
    return analyzer.analyze_texts(text_dict,metric_weights)
    
def test_analyze_texts():
    examples = {
        1: """
        The sun casts a golden hue over the horizon, painting the sky with shades of orange and pink. Birds sing melodiously, their tunes harmonizing with the rustling of leaves. The meadow is dotted with wildflowers, each swaying gently in the morning breeze. A serene river flows nearby, its surface reflecting the clear blue sky and lush green trees lining its banks. Every element of nature seems to be in perfect harmony, creating an ambiance of peace and tranquility.
        """,
        
        2: """
        Jane walked hurriedly through the streets of the old town, her heels clicking against the cobblestone roads. The alleys were narrow, lined with ancient brick buildings with wooden shutters. She could hear distant laughter from a nearby pub and the faint strumming of a guitar. As she turned a corner, she bumped into a tall man with piercing blue eyes. Their gaze met, and in that fleeting moment, a story of love, betrayal, and redemption began to unfold.
        """,
        
        3: """
        The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system. Comprising over 2,900 individual reefs and 900 islands, it stretches for over 2,300 kilometers. Rich in marine diversity, the reef is home to countless species of fish, mollusks, and starfish, as well as turtles and dolphins. A UNESCO World Heritage site, it plays a crucial role in marine ecology and is a popular destination for tourists and divers. However, rising ocean temperatures and pollution pose a significant threat to its survival.
        """,
        
        4: """
        User: Hello, ChatGPT! Can you help me with some information on black holes?
        ChatGPT: Of course! Black holes are regions in space where the gravitational pull is so strong that nothing, not even light, can escape from it. They are formed when massive stars collapse at the end of their life cycles. What else would you like to know?
        User: That's fascinating! How do we detect them?
        ChatGPT: Black holes can't be observed directly due to their strong gravitational pull. However, they can be detected by observing the effect of their gravitational forces on nearby celestial objects and light. For example, when matter is drawn towards a black hole, it heats up and emits X-rays, which can be detected.
        User: Wow, thanks for the explanation!
        ChatGPT: You're welcome! If you have any more questions, feel free to ask.
        """,
        5: """
        The sun casts a golden hue over the horizon, painting the sky with shades of orange and pink. Birds sing melodiously, their tunes harmonizing with the rustling of leaves. The meadow is dotted with wildflowers, each swaying gently in the morning breeze. A serene river flows nearby, its surface reflecting the clear blue sky and lush green trees lining its banks. Every element of nature seems to be in perfect harmony, creating an ambiance of peace and tranquility.
        """
        
    }

    metric_weights = {
        "Normalized Unigram Diversity": 0.8,  # Diversity is crucial for synthetic data to ensure it's not overly repetitive.
        "Normalized Bigram Diversity": 0.9,   # Bigrams give a more contextual diversity check than unigrams.
        "Normalized Trigram Diversity": 1.0,  # Trigrams are even more contextual; hence the highest weight among n-grams.
        "TTR": 0.7,                           # Type-Token Ratio gives a sense of vocabulary richness.
        "Local Coherence": 1.2,               # Ensures the text flows well and makes contextual sense.
        "Reference Resolution": 1.1,          # Ensures that pronouns and other referring expressions link correctly.
        "Lexical Density": 0.7,               # High lexical density might indicate good information content.
        "Hapax Legomena": 0.6,                # Rare words can add richness but also can be a sign of overfitting or noise.
        "Average Sentence Length": 0.5,       # Very long sentences can be hard to follow, but this metric should not be heavily penalized.
        "Readability Score": 1.0,             # Ensures the text is easily understandable.
        "Syntactic Complexity": 0.8,          # A good balance of syntactic structures can enhance text quality.
        "Semantic Depth": 0.9,                # Ensuring named entities and other semantically rich elements are present.
        "positive": 0.5,                      # Sentiment scores might not be too crucial unless you're targeting a specific sentiment.
        "negative": 0.5,
        "neutral": 0.4
    }


    analysis_results, ranked_texts, sentence_similarity = analyze_texts(examples, metric_weights)
    print(f"RANK SORTED OF SENTANCES: {ranked_texts}")
    for key, value in analysis_results.items():
        for metric, score in value.items():
            if metric != "Original Text":
                print(key, metric, score)

    # Print sentence similarity results
    print("\nSentence Similarity Results:")
    for text_id, similarities in sentence_similarity.items():
        print(f"Text ID: {text_id}")
        for pair, similarity in similarities.items():
            print(pair, similarity)
        print("---")
# Running the test function
test_analyze_texts()
