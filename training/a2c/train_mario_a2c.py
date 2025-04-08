import gymnasium as gym
import ale_py
import sys
from stable_baselines3 import A2C

gym.register_envs(ale_py)

env = gym.make("ALE/MarioBros-v5")

model = A2C("MlpPolicy", env, verbose =1, tensorboard_log="./logs/mario/")
model.learn(total_timesteps=int(sys.argv[1]) , progress_bar=True)
model.save(sys.argv[2])