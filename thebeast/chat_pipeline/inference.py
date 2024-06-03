from openai import OpenAI
import openai
import time
import multiprocessing
import random
import uuid

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
                        # Generate a random configuration for the response
                        max_tokens = random.choice(range(model_config.get('max_tokens_min', 100), model_config.get('max_tokens_max', 1000) + 1, 10))
                        temperature = round(random.uniform(model_config.get('temperature_min', 0.7), model_config.get('temperature_max', 1.0)), 2)
                        top_p = round(random.uniform(model_config.get('top_p_min', 0.9), model_config.get('top_p_max', 1.0)), 2)

                        # API call
                        response = self.client.chat.completions.create(
                            model=model_config.get('name', "gpt-4o"),
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": input_text}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=model_config.get('frequency_penalty', 0.0),
                            presence_penalty=model_config.get('presence_penalty', 0.0),
                            logprobs=True,
                            top_logprobs=5
                        )

                        # Add UUID and configuration details to the response
                        response_data = {
                            'uuid': str(uuid.uuid4()),  # Generate a unique identifier
                            'response': response,
                            'configuration': {
                                'max_tokens': max_tokens,
                                'temperature': temperature,
                                'top_p': top_p,
                                'model': model_config.get('name', "gpt-4o")
                            }
                        }
                        responses.append(response_data)
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

    def extract_chat_completion_data(self, responses):
        response_data_list = []
        for response_data in responses:
            if response_data['response'].choices:
                choice = response_data['response'].choices[0]
                response_content = choice.message.content if hasattr(choice.message, 'content') else ""
                response_data_list.append({
                    "uuid": response_data['uuid'],
                    "response_content": response_content,
                    "configuration": response_data['configuration']
                })
        return response_data_list




    


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
