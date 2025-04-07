import gymnasium as gym
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make(sys.argv[1], render_mode="rgb_array")
model = DQN.load(sys.argv[2], env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")