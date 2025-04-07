import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

env = gym.make("Taxi-v3")
q_table = np.zeros((env.observation_space.n, env.action_space.n))

#hyperparameters
epsilon = 0.5
max_epsilon = 1
min_epsilon = 0.1
decay = 0.01
learning_rate = 0.7
discount_factor = 0.618
max_episodes = int(sys.argv[1])
max_steps = 100
training_rewards = []
total_training_rewards = 0

state, info = env.reset(seed=42)

for episode in range(max_episodes):
    state, info = env.reset()
    for step in range(max_steps):
        #generate a random number and see if it passes the epsilon threshold
        if random.uniform(0, 1) < epsilon:
            #pick a random action
            action = env.action_space.sample()
        else:
            #pick the action with the most value on that state in the q_table
            action = np.argmax(q_table[state, :])

        new_state, reward, done, truncated, info = env.step(action)

        #update state and q_table item
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :])-q_table[state, action])
        state = new_state
        total_training_rewards += reward

        #check if the episode is done
        if done or truncated:
            break
    #decay epsilon threshold
    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
    training_rewards.append(total_training_rewards)
env.close()

#Finished Model Test
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, truncated, _ = env.step(action)
env.close()

#Plot results
x = range(max_episodes)
plt.plot(x, training_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.gca().invert_yaxis()
plt.show()
