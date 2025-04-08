import gymnasium as gym
import ale_py
import sys
import stable_baselines3
from stable_baselines3 import PPO

gym.register_envs(ale_py)

env = gym.make("ALE/Tetris-v5", render_mode = "human")

model = PPO("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=10000 , progress_bar=True)
model.save("a2c.tetris")


