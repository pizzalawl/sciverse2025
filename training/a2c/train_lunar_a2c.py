import gymnasium as gym
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/lunar/")
model.learn(total_timesteps=int(sys.argv[1]), progress_bar=True)
model.save(sys.argv[2])