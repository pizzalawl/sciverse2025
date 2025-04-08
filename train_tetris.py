import gymnasium as gym
import ale_py
import sys
from stable_baselines3 import A2C

gym.register_envs(ale_py)

env = gym.make("ALE/Tetris-v5", render_mode = "human")

model = A2C("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=sys.argv[1] , progress_bar=True)
model.save("a2c_tetris.zip")


