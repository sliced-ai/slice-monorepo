from openai import OpenAI

API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

class InferenceEngine:
    def __init__(self, models_config):
        self.models_config = models_config
        self.client = OpenAI(api_key=API_KEY)

    def generate_responses(self, input_text):
        all_responses = []
        for model_config in self.models_config:
            responses = []
            for _ in range(model_config.get('n', 1)):
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Replace with your model ID as needed
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text}
                    ],
                    max_tokens=model_config.get('max_tokens', 1000),
                    temperature=model_config.get('temperature', 0.7),
                    top_p=model_config.get('top_p', 1.0),
                    frequency_penalty=model_config.get('frequency_penalty', 0.0),
                    presence_penalty=model_config.get('presence_penalty', 0.0),
                    logprobs=True,  # Enable logprobs
                    top_logprobs=5  # Specify number of top log probabilities to return
                )
                responses.append(response)  # Accumulating responses for each iteration
    
            processed_responses = [self.extract_chat_completion_data(response) for response in responses]
            all_responses.append(processed_responses)
        return all_responses

    def get_full_response(self, prompt, n=1, max_tokens=1000, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        for _ in range(n):
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Replace with your model ID as needed
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=True,  # Enable logprobs
                top_logprobs=10  # Specify number of top log probabilities to return
            )
    
            return response

    def extract_chat_completion_data(self, response):
        # Data structure to hold the results
        data = {
            "Response Content": "",
            "Logprobs": [],
            "Top Logprob Words": [],
            "Top Logprob Values": []
        }
        
        # Assume the first choice for simplification; adapt as needed for multiple choices
        if response.choices:
            choice = response.choices[0]
            data["Response Content"] = choice.message.content
            
            # Extract token logprob information
            for token_logprob in choice.logprobs.content:
                # Append the logprob of the current token to the list
                data["Logprobs"].append(token_logprob.logprob)
                
                # For collecting top logprob words and their values
                top_words = []
                top_values = []
                
                # Extract top logprob details
                for top_logprob in token_logprob.top_logprobs:
                    top_words.append(top_logprob.token)
                    top_values.append(top_logprob.logprob)
                
                # Append each token's top logprob words and values
                data["Top Logprob Words"].append(top_words)
                data["Top Logprob Values"].append(top_values)
    
        return data


# Testing the modified InferenceEngine
if __name__ == "__main__":
    engine = InferenceEngine([
        {"name": "gpt-4o", "n": 2, "max_tokens": 150, "temperature": 0.7, "top_p": 0.9},
        {"name": "gpt-4o", "n": 3, "max_tokens": 200, "temperature": 0.6, "top_p": 1.0}
    ])
    
    test_input = "How does quantum computing work?"
    results = engine.generate_responses(test_input)
    for result in results:
        print(result)
