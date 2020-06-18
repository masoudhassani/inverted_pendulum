from __future__ import print_function
from environment import Environment
import random
import numpy as np
import math
import re

import matplotlib
matplotlib.use('Qt4Agg',warn=False, force=True)
from matplotlib import pyplot as plt
# gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']

# parameters
max_episodes = 20001
stacking_steps = 1
num_steps = 1200/stacking_steps   # 20 sec
num_steps_decay = 0.999
min_num_steps = 1200/stacking_steps   # 20 sec
epsilon = 1
epsilon_upright = 1
max_epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.0001
discount_factor = 0.8
learning_rate = 0.99
learning_rate_decay = 1.0
min_learning_rate = 0.01
upright = False
threshold = 0.174533  # 10 deg, smaller than this angle is considered upward

save_every = 1000
print_every = 100
render_real_time_every = 250
PI = 3.14159265359

# initialize the environment
env = Environment(render=False, pendulum_length=0.12, stacking_steps=stacking_steps) 
print('state space:', env.state_space_size)
print('action space:', env.action_space_size)

# initialization for plotting 
plt.ion()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.grid(True)
plt.show()
episode_rewards = []

# uncomment the following to initialize the q table
q_table_size = (env.encoded_state_size, env.action_space_size)
q_table = np.zeros(q_table_size)
min_episode = 0

# uncomment the following the read the q table from a numpy array
# q_table_file = 'models/q_table_ep10000.npy'
# q_table = np.load(q_table_file)
# epsilon = min_epsilon
# epsilon_upright = 0.00157
# min_episode = int(re.findall(r'\d+', q_table_file)[0]) + 1

print('q_table shape:', q_table.shape)

for e in range(min_episode, max_episodes):
    # switch between real time and free running
    real_time = False
    if e % render_real_time_every == 0:
        real_time = True

    # reset the environment
    raw_state, current_state = env.reset(real_time)

    # stabilizing the simulator at the satrt of each episode
    for i in range(20/stacking_steps):
        _, _, current_state, _ = env.step(1)

    step = 0
    episode_reward = 0 
    reward = 0
    
    # decay epsilon for upright situation
    if upright:
        epsilon_upright = env.parameter_decay(epsilon_upright, min_epsilon, epsilon_decay, type='linear')
    upright = False

    while True:
        step += 1
        if step > num_steps:
            break

        # use a different epsilon if the pundulum is upright
        if raw_state[-2] >= 0:
            diff = abs(-PI + raw_state[-2])
        else:
            diff = abs(PI + raw_state[-2])  
        
        if diff <= threshold:
            eps = epsilon_upright
            upright = True
        else:
            eps = epsilon

        # epsilon greedy algorithm
        if np.random.random() > eps:
            action = np.argmax(q_table[current_state])

        else:
            action = env.sample()
        
        # take an action and read the consequent state
        raw_state, _, new_state, reward = env.step(action)

        # max possible q value of the next state
        max_future_q = np.max(q_table[new_state])

        # q value of the state/action combination
        current_q = q_table[current_state][action]

        # update the q value for state/action combination
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[current_state][action] = new_q

        # set the captured state as the current state
        current_state = new_state

        # update episode reward
        episode_reward += reward
    
    # add the cuurent episode reward to plot
    episode_rewards.append(episode_reward)

    # update epsilon
    epsilon = env.parameter_decay(epsilon, min_epsilon, epsilon_decay, type='linear')

    # update learning rate
    learning_rate = env.parameter_decay(learning_rate, min_learning_rate, learning_rate_decay, type='linear')

    # update number of steps
    num_steps = max(min_num_steps, int(num_steps*num_steps_decay))

    if e % save_every == 0:
        np.save('q_table_ep{}.npy'.format(e), q_table)
        
    print('episode: ', e, 'epsilon:', epsilon, 'epsilon_upright:', epsilon_upright, 'learning_rate:', learning_rate, 'max steps:', num_steps, 'episode_reward:', episode_reward)


plt.plot(episode_rewards, 'b-')
plt.draw()
plt.pause(0.1)
plt.savefig('reward_episode_{}.png'.format(e))
print(q_table)
print(sum(episode_rewards))
