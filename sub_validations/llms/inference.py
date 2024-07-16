import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import random
import multiprocessing
import pynvml
import time
import matplotlib.pyplot as plt

class DistributedInference:
    def __init__(self, model_path, gpus, prompt, total_inferences, temperature_range, top_p_range, max_new_tokens_range):
        self.model_path = model_path
        self.gpus = gpus
        self.prompt = prompt
        self.total_inferences = total_inferences
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_new_tokens_range = max_new_tokens_range
        self.inferences_per_gpu = total_inferences // len(gpus)
        self.manager = multiprocessing.Manager()
        self.return_dict = self.manager.dict()

    @staticmethod
    def monitor_gpu_utilization(gpu_index, duration, interval, return_dict):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        utilization = []
        for _ in range(int(duration / interval)):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            utilization.append(util)
            time.sleep(interval)
        pynvml.nvmlShutdown()
        return_dict[gpu_index] = utilization

    @staticmethod
    def run_inference(model_path, device, device_index, prompt, n_inferences, temperature_range, top_p_range, max_new_tokens_range, return_dict):
        # Initialize the tokenizer and model on the specific device
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": device}, torch_dtype=torch.bfloat16)

        # Initialize the pipeline
        generate_text = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device}
        )

        # Start GPU monitoring
        monitor_duration = 60  # Monitor duration in seconds
        monitor_interval = 1   # Monitor interval in seconds
        gpu_util_dict = multiprocessing.Manager().dict()
        monitor_process = multiprocessing.Process(target=DistributedInference.monitor_gpu_utilization, args=(device_index, monitor_duration, monitor_interval, gpu_util_dict))
        monitor_process.start()

        # Generate and print the responses
        for i in range(n_inferences):
            temperature = random.uniform(*temperature_range)
            top_p = random.uniform(*top_p_range)
            max_new_tokens = random.randint(*max_new_tokens_range)
            
            generate_kwargs = {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1
            }
            
            result = generate_text(prompt, **generate_kwargs)
            
            print(f"Device: {device}, Inference {i+1}:")
            print(f"Temperature: {temperature}")
            print(f"Top_p: {top_p}")
            print(f"Max_new_tokens: {max_new_tokens}")
            print(f"Generated Text: {result[0]['generated_text']}")
            print("-" * 50)
        
        monitor_process.join()
        return_dict[device_index] = gpu_util_dict[device_index]

    def run(self):
        # Set the multiprocessing start method to 'spawn' for compatibility
        multiprocessing.set_start_method('spawn', force=True)

        # Create a list of processes
        processes = []
        
        for idx, gpu in enumerate(self.gpus):
            p = multiprocessing.Process(target=DistributedInference.run_inference, args=(
                self.model_path, gpu, idx, self.prompt, self.inferences_per_gpu, self.temperature_range, self.top_p_range, self.max_new_tokens_range, self.return_dict))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Aggregate GPU utilization data
        all_utilization = []
        for key in self.return_dict.keys():
            all_utilization.extend(self.return_dict[key])
        
        # Compute average utilization
        avg_utilization = [sum(util) / len(self.gpus) for util in zip(*[self.return_dict[i] for i in range(len(self.gpus))])]

        # Plot the average GPU utilization
        plt.figure(figsize=(10, 5))
        plt.plot(avg_utilization, label='Average GPU Utilization')
        plt.xlabel('Time (s)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('Average GPU Utilization Over Time')
        plt.legend()
        plot_path = 'gpu_utilization_plot.png'
        plt.savefig(plot_path)
        
        print(f"GPU utilization plot saved to {plot_path}")

# Usage Example
if __name__ == "__main__":
    # Define the parameters
    model_path = "EleutherAI/pythia-70m"
    gpus = ["cuda:0"]
    prompt = "Explain to me the difference between nuclear fission and fusion."
    total_inferences = 10
    temperature_range = (0.5, 1.0)
    top_p_range = (0.8, 1.0)
    max_new_tokens_range = (50, 300)

    # Create and run the distributed inference
    inference = DistributedInference(model_path, gpus, prompt, total_inferences, temperature_range, top_p_range, max_new_tokens_range)
    inference.run()
