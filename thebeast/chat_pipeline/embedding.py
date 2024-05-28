from openai import OpenAI

class EmbeddingSystem:
    
    def __init__(self, api_key, embedding_models_config):
        self.client = OpenAI(api_key=api_key)
        self.embedding_models_config = embedding_models_config

    def create_embeddings(self, texts):
        embeddings = {}
        for model_config in self.embedding_models_config:
            model_embeddings = []
            for text in texts:
                embedding = self.get_embedding(text, model=model_config['model'])
                model_embeddings.append(embedding)
            embeddings[model_config['name']] = model_embeddings
        return embeddings

    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

def main():

    API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'
    embedding_models_config = [
        {'name': 'default_model', 'model': 'text-embedding-3-large'},
        {'name': 'small_model', 'model': 'text-embedding-3-small'}
    ]

    # Initialize the embedding system with API key and configurations
    embedding_system = EmbeddingSystem(api_key=API_KEY,
            embedding_models_config=embedding_models_config)

    # Sample texts to generate embeddings
    test_texts = ["Hello world!", "How are you doing today?"]

    # Create embeddings
    all_embeddings = embedding_system.create_embeddings(test_texts)

    # Print results
    for model_name, embeddings in all_embeddings.items():
        print(f"Embeddings for {model_name}:")
        for i, embedding in enumerate(embeddings):
            print(f"Text: {test_texts[i]}, Embedding: {embedding}")

if __name__ == '__main__':
    main()
