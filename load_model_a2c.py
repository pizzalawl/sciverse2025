import sys
import gymnasium as gym
from stable_baselines3 import A2C
import ale_py

gym.register_envs(ale_py)
env = gym.make(sys.argv[1], render_mode="human")

model = A2C.load(sys.argv[2], env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(int(sys.argv[3])):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")