import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3")
q_table = np.zeros((env.observation_space.n, env.action_space.n))
print(env.observation_space.n)

#hyperparameters
epsilon = 0.5
max_epsilon = 1
min_epsilon = 0.1
decay = 0.01
learning_rate = 0.7
discount_factor = 0.618
max_episodes = 1000
max_steps = 100

state, info = env.reset(seed=42)

for episode in range(max_episodes):
    state, info = env.reset()
    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, done, truncated, info = env.step(action)

        #update epsilon, state, and q_table
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :])-q_table[state, action])
        state = new_state

        print(reward)

        if done or truncated:
            break
    print("episode done")
    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
env.close()

#Finished Model Test
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, truncated, _ = env.step(action)
env.close()