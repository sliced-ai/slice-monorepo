import gym
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained 70M Pythia model and tokenizer from Hugging Face
model_name = "EleutherAI/pythia-70m-deduped"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Initialize the CartPole environment
env = gym.make("CartPole-v1")

# Function to preprocess the observation
def preprocess_observation(obs):
    obs_str = " ".join(map(str, obs))
    inputs = tokenizer(obs_str, return_tensors="pt")
    return inputs

# Function to choose an action based on the model's output
def choose_action(outputs):
    logits = outputs.logits[0, -1, :]
    action = torch.argmax(logits).item() % env.action_space.n
    return action

# RL loop
num_episodes = 10

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    
    for step in range(1000):
        env.render()

        # Preprocess the observation
        inputs = preprocess_observation(obs)

        # Get the model's output
        with torch.no_grad():
            outputs = model(**inputs)

        # Choose an action based on the model's output
        action = choose_action(outputs)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
