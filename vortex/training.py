from environment import Environment
import random

# parameters
max_episodes = 10
max_steps = 600
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01
gamma = 0.99

# initialize the environment
env = Environment()
print(env.state_space_size)
print(env.action_space_size)

for e in range(max_episodes):
    current_state = env.reset()
    step = 0
    episod_reward = 0 

    while True:
        step += 1
        if step > max_steps:
            break

        # epsilon greedy algorithm
        if random.random() > epsilon:
            pass
        else:
            action = env.sample()
            state, reward = env.step(action)

        # update episod reward
        episod_reward += reward

    # update epsilon
    epsilon = max(min_epsilon, epsilon*epsilon_decay)