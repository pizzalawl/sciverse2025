import gymnasium as gym
from stable_baselines3 import PPO
import ale_py
import sys

gym.register_envs(ale_py)
env = gym.make(sys.argv[1], render_mode="human")

model = PPO.load(sys.argv[2], env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(sys.argv[3]):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")