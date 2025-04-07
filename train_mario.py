import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(5000), progress_bar=True)
model.save("dqn_lunar")