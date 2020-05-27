from environment import Environment
import random
import numpy as np

# parameters
max_episodes = 10001
max_steps = 600
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
discount_factor = 0.99
learning_rate = 0.01
learning_rate_decay = 0.99
min_learning_rate = 0.001

save_every = 2000

# initialize the environment
env = Environment()
print(env.state_space_size)
print(env.action_space_size)

# initialize the q table
q_table_size = env.state_space_size
q_table_size.append(env.action_space_size)

q_table = np.zeros(q_table_size)
# q_table = np.random.uniform(low=-2, high=0, size=(env.state_space_size + [env.action_space_size]))
print(q_table.shape)

for e in range(max_episodes):
    current_state = env.reset()
    step = 0
    episod_reward = 0 
    reward = 0
    done = False

    while not done:
        step += 1
        if step > max_steps:
            break

        # epsilon greedy algorithm
        if np.random.random() > epsilon:
            ind = np.argmax(q_table[current_state])
            action = ind - int(ind/env.action_space_size)*env.action_space_size
            
            # print(np.max(q_table[current_state]))
            # print(q_table[current_state])
            # print(current_state)
            # print(ind)
            # print(action)

        else:
            action = env.sample()

        # update q_table
        new_state, reward = env.step(action)
        max_future_q = np.max(q_table[new_state])
        current_q = q_table[current_state + [action, ]]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[current_state + [action, ]] = new_q
        current_state = new_state

        # update episod reward
        episod_reward += reward

    # update epsilon
    epsilon = max(min_epsilon, epsilon*epsilon_decay)

    # update learning rate
    learning_rate = max(min_learning_rate, learning_rate*learning_rate_decay)


    if e % save_every == 0:
        np.save('q_table_ep{}.npy'.format(e), q_table)
        
    print('episode: ', e, 'epsilon:', epsilon, 'learning_rate:', learning_rate, 'episod_reward:', episod_reward)

print(q_table)