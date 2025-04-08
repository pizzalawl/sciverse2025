import gymnasium as gym
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/lunar/")
model.learn(total_timesteps=int(sys.argv[1]), progress_bar=True)
model.save("pretrained/lunar/ppo_lunar.zip")