import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from cuml.manifold import UMAP as cumlUMAP
import matplotlib.pyplot as plt
import numpy as np

# Define the Autoencoder model
class BertAutoencoder(nn.Module):
    def __init__(self, bert_model_name, lstm_units=256, max_length=512):
        super(BertAutoencoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.encoder = self.bert.encoder
        self.decoder = nn.LSTM(self.bert.config.hidden_size, lstm_units, batch_first=True)
        self.output_layer = nn.Linear(lstm_units, self.bert.config.vocab_size)
        self.max_length = max_length

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            encoder_outputs = bert_outputs.last_hidden_state
        return encoder_outputs

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertAutoencoder('bert-base-uncased').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Function to process a JSON file and extract the relevant data
def extract_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    
    extracted_data = []
    for document in data:
        for turn in document['TURNS']:
            for name in turn['NAMES']:
                extracted_data.append({
                    'name': name,
                    'utterance': ' '.join(turn['UTTERANCES']),
                    'turn_number': turn['NUMBER']
                })
    return extracted_data

# Load data from JSON files in the specified folder and its subdirectories
def load_data(folder_path, max_length=None):
    all_texts = []
    all_names = []
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                data = extract_data(file_path)
                for item in data:
                    if max_length is None or len(item['utterance']) <= max_length:
                        all_texts.append(item['utterance'])
                        all_names.append(item['name'])
                file_count += 1
    print(f"Processed {file_count} files and extracted {len(all_texts)} utterances suitable within length {max_length}.")
    return all_texts, all_names

def preprocess_texts(texts, tokenizer, max_length=128):
    if not texts:  # Check if texts is empty
        print("No texts available for tokenization.")
        return None, None
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings.input_ids, encodings.attention_mask

def visualize_embeddings(embeddings, labels, title='Embeddings', file_name='embeddings.png'):
    plt.figure(figsize=(10, 7))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for idx, label in enumerate(unique_labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors(idx), label=label, alpha=0.5)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1), title='Names')
    plt.savefig(file_name)
    plt.show()

def main():
    folder_path = '/workspace/slice-monorepo/cl_cr3/aligneddata'  # Update with your actual data path
    max_text_length = 150  # Set the maximum text length you want to include
    batch_size = 8192  # Define the batch size for processing
    texts, names = load_data(folder_path, max_text_length)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = load_model('trained_autoencoder.pth')
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        input_ids, attention_mask = preprocess_texts(batch_texts, tokenizer)
        if input_ids is None:
            continue
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        embeddings = model.encode(input_ids, attention_mask)
        embeddings = embeddings[:, 0, :].cpu().numpy()  # Taking the embedding of the [CLS] token
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)

    # Dimensionality reduction with GPU UMAP
    reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(all_embeddings)

    # Visualize and save UMAP embeddings
    visualize_embeddings(umap_embeddings, names, 'UMAP Embeddings', 'umap_embeddings.png')

if __name__ == "__main__":
    main()
