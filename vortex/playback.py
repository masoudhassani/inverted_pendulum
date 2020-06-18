from __future__ import print_function
from environment import Environment
import random
import numpy as np
import matplotlib.pyplot as plt 
import math
import re

# load a trained model
# do not forget to unzip if necessary
q_table_file = 'models/q_table_ep17000.npy'
q_table = np.load(q_table_file)
episode = int(re.findall(r'\d+', q_table_file)[0])
stacking_steps = 1
real_time = True

# initialize the environment
env = Environment(render=True, pendulum_length=0.12, stacking_steps=stacking_steps) 
episode_reward = 0

# reset the environment
_, current_state = env.reset(real_time)
env.set_text('Agent learns how to stabilize')
# env.set_text('Episode {}'.format(episode))

for i in range(2000):
    # select an action based on the current state
    action = np.argmax(q_table[current_state])

    # take ac action and read the consequent state
    _, _, new_state, reward = env.step(action)    

    current_state = new_state
    episode_reward += reward

    print ('episod reward:', episode_reward, 'state reward:', reward)