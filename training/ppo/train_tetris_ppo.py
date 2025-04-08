import gymnasium as gym
import ale_py
import sys
from stable_baselines3 import PPO
gym.register_envs(ale_py)

env = gym.make("ALE/Tetris-v5")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tetris/")
model.learn(total_timesteps=int(sys.argv[1]) , progress_bar=True)
model.save(sys.argv[2])