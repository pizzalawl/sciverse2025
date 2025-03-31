import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

#environment setup
env = gym.make("Taxi-v3", render_mode="rgb_array")
observation, info = env.reset()
np.random.seed(0) 

#hyperparameters
q_table = np.zeros((env.observation_space.n, env.action_space.n))
print(f"Table is {env.observation_space.n}*{env.action_space.n}")
epsilon = 0.1
learning_rate = 0.7               
discount_factor = 0.618
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
epochs = 0
episodes = 2000
done = False

while not done:
    #get epsilon-greedy value and inital state
    random_epsilon = random.uniform(0, 1)
    state = env.reset()

    #check if to do action based on exploration or exploitation
    if random_epsilon < epsilon and epochs != 0:
        #pick the action of the current state with the highest q-value
        action = np.argmax(q_table[state[0], :])
        new_state, reward, done, truncated, info = env.step(action)
    else:
        #pick a random action
        action = env.action_space.sample()
        new_state, reward, done, truncated, info = env.step(action)

    #update q-table
    print(f"Changing value: {state[0]}*{action}")
    q_table[state[0], action] = q_table[state[0], action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state[0], action])
    print(f"New value is: {q_table[state[0], action]}")

    #update epoch counter
    epochs += 1
    if epochs == episodes:
        break

print("Final Q_Table is:")
for x in q_table:
    for y in x:
        print(" " + str(y), end="")
    print("\n")