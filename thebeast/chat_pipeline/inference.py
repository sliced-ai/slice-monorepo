# inference_engine.py
class InferenceEngine:
    def __init__(self, models_config):
        self.models_config = models_config

    def generate_responses(self, input_text):
        responses = []
        for model_config in self.models_config:
            response = self.fake_inference_function(input_text, model_config)
            responses.append(response)
        return responses

    def fake_inference_function(self, input_text, model_config):
        return f"Simulated response from {model_config['name']} for text '{input_text}'"
