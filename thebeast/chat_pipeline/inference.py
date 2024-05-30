from openai import OpenAI
import openai
import time
import multiprocessing
import random

API_KEY = 'sk-proj-7MAfZbOm9lPY28pubTiRT3BlbkFJGgn73o5e6sVCjoTfoFAP'

class InferenceEngine:
    def __init__(self, models_config):
        self.models_config = models_config
        self.client = OpenAI(api_key=API_KEY)
    
    def generate_responses(self, input_text):
        def worker(model_config, n_responses, input_text, return_dict, index):
            responses = []
            for _ in range(n_responses):
                while True:
                    try:
                        response = self.client.chat.completions.create(
                            model=model_config.get('model', "gpt-4o"),
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": input_text}
                            ],
                            max_tokens=model_config.get('max_tokens', 1000),
                            temperature=model_config.get('temperature', 0.7),
                            top_p=model_config.get('top_p', 1.0),
                            frequency_penalty=model_config.get('frequency_penalty', 0.0),
                            presence_penalty=model_config.get('presence_penalty', 0.0),
                            logprobs=True,
                            top_logprobs=5
                        )
                        responses.append(response)
                        break
                    except openai.RateLimitError as e:
                        wait_time = random.uniform(1, 300)  # Randomized wait time
                        print(f"Rate limit hit. Process {index} waiting for {wait_time} seconds. Error: {e}")
                        time.sleep(wait_time)
                    except openai.APIError as e:
                        print(f"OpenAI API returned an API Error: {e}")
                        break
                    except openai.APIConnectionError as e:
                        print(f"Failed to connect to OpenAI API: {e}")
                        time.sleep(2)
                    except Exception as e:
                        print(f"Unexpected error in process {index}: {e}")
                        break
            return_dict[index] = responses
            print(f"Process {index} completed with {n_responses} responses.")
    
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
    
        total_responses = sum(model_config.get('n', 1) for model_config in self.models_config)
        max_responses_per_process = 5  # Set the max responses per process
        num_processes = (total_responses + max_responses_per_process - 1) // max_responses_per_process
    
        print(f"Total responses to generate: {total_responses}")
        print(f"Running inference with {num_processes} processes.")
    
        current_process = 0
        for model_config in self.models_config:
            n_responses = model_config.get('n', 1)
            while n_responses > 0:
                responses_for_this_process = min(n_responses, max_responses_per_process)
                n_responses -= responses_for_this_process
    
                p = multiprocessing.Process(target=worker, args=(model_config, responses_for_this_process, input_text, return_dict, current_process))
                jobs.append(p)
                p.start()
                current_process += 1
    
        for proc in jobs:
            proc.join()
    
        all_responses = []
        for responses in return_dict.values():
            all_responses.extend(responses)
    
        print(f"Total responses generated: {len(all_responses)}")
        return all_responses

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
