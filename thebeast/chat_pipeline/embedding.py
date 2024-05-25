# embedding_system.py
class EmbeddingSystem:
    def __init__(self, embedding_models_config):
        self.embedding_models_config = embedding_models_config

    def create_embeddings(self, texts):
        embeddings = {}
        for model_config in self.embedding_models_config:
            model_embeddings = []
            for text in texts:
                embedding = self.fake_embedding_function(text, model_config)
                model_embeddings.append(embedding)
            embeddings[model_config['name']] = model_embeddings
        return embeddings

    def fake_embedding_function(self, text, model_config):
        return f"Simulated embedding from {model_config['name']} for text '{text}'"
