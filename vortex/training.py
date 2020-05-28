from __future__ import print_function
from environment import Environment
import random
import numpy as np
import matplotlib.pyplot as plt 


# parameters
max_episodes = 10001
num_steps = 4800   # 80 sec
num_steps_decay = 0.999
min_num_steps = 600   # 10 sec
epsilon = 1.0
epsilon_decay = 0.998
min_epsilon = 0.01
discount_factor = 0.99
learning_rate = 0.02
learning_rate_decay = 0.99
min_learning_rate = 0.001

save_every = 500
print_every = 100
render_real_time_every = 250

# initialize the environment
env = Environment()
print(env.state_space_size)
print(env.action_space_size)

# initialization for plotting 
# plt.ion()
fig=plt.figure()
episod_rewards = []

# initialize the q table
q_table_size = (env.encoded_state_size, env.action_space_size)

q_table = np.zeros(q_table_size)
# q_table = np.random.uniform(low=-2300, high=-1500, size=q_table_size)
print('q_table shape:', q_table.shape)

for e in range(max_episodes):
    # switch between real time and free running
    real_time = False
    if e % render_real_time_every == 0:
        real_time = True

    # reset the environment
    _, current_state = env.reset(real_time)

    # stabilizing the simulator at the satrt of each episode
    for i in range(50):
        _, current_state, _ = env.step(2)

    step = 0
    episod_reward = 0 
    reward = 0
    while True:
        step += 1
        if step > num_steps:
            break

        # epsilon greedy algorithm
        if np.random.random() > epsilon:
            action = np.argmax(q_table[current_state])
            if e % print_every == 0:
                print('agent action:', action, end='\r')

        else:
            action = env.sample()
            # if e % print_every == 0:
                # print('randm action:', action)

        # take ac action and read the consequent state
        _, new_state, reward = env.step(action)

        # max possible q value of the next state
        max_future_q = np.max(q_table[new_state])

        # q value of the state/action combination
        current_q = q_table[current_state][action]

        # update the q value for state/action combination
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[current_state][action] = new_q

        # set the captured state as the current state
        current_state = new_state

        # update episod reward
        episod_reward += reward
    
    # add the cuurent episode reward to plot
    episod_rewards.append(episod_reward)

    # update epsilon
    epsilon = max(min_epsilon, epsilon*epsilon_decay)

    # update learning rate
    learning_rate = max(min_learning_rate, learning_rate*learning_rate_decay)

    # update number of steps
    num_steps = max(min_num_steps, int(num_steps*num_steps_decay))

    if e % save_every == 0:
        np.save('q_table_ep{}.npy'.format(e), q_table)
        
    print('episode: ', e, 'epsilon:', epsilon, 'learning_rate:', learning_rate, 'max steps:', num_steps, 'episod_reward:', episod_reward)

print(q_table)
plt.plot(episod_rewards)
plt.draw() 
plt.pause(5)