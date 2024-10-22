import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create a vectorized environment
env_id = "CartPole-v1"
num_envs = 64  # Number of parallel environments
env = make_vec_env(env_id, n_envs=num_envs)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")

# Define a callback for evaluation
eval_env = gym.make(env_id)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=1000000, callback=eval_callback)

# Save the model
model.save("ppo_cartpole")

# Load the model
model = PPO.load("ppo_cartpole")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
