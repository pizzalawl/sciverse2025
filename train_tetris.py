import gymnasium as gym
import ale_py
import stable_baselines3
from stable_baselines3 import A2C

gym.register_envs(ale_py)

env = gym.make("ALE/Tetris-v5", render_mode = "human")

model = A2C("MlpPolicy", env, verbose =1)
model.learn(total_timesteps= 1000)


episodes = 10

for ep in range(episodes):
   obs = env.reset()
   done = False
   while not done:
      env.render()
      env.step(env.action_space.sample())

env.close()

