from __future__ import print_function
from environment import Environment
import random
import numpy as np
import math
import re
import time
import logging 
import cntk as ck 
import matplotlib
# matplotlib.use('Qt4Agg',warn=False, force=True)
from matplotlib import pyplot as plt
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
import sys
sys.path.insert(0,'..')
from modules.dqn_agent import Agent 

PI = 3.14159265359

### SETUP GPU (TENSORFLOW ONLY) ###############################
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
###############################################################

# LOGGING CONFIG ##############################
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
###############################################

# TRAINING PARAMETERS #########################
render = False              # if true, the gameplay will be shown
render_every = 1            # every n episode, render real time if render is True
save_every = 200            # frequency of saving the model weights, number of episodes
aggregate_stats_every = 5   # every n episode 
max_steps = 600             # number of time steps per game
stack_size = 4              # number of consecutive frames to be stacked to represent a state
max_episodes = 2001        # number of games to play
min_reward_save = 10000       # min reward threshold for saving a model weight
model_name = '24x24'
last_episode = 0
discount_factor = 0.95
learning_rate = 0.001
upright = False
threshold = 0.174533  # 10 deg, smaller than this angle is considered upward
save_every = 1000
print_every = 100
render_real_time_every = 250
last_episode = 0
# epsilon = 1
# epsilon_upright = 1
# max_epsilon = 1.0
# epsilon_decay = 0.999
# min_epsilon = 0.0001
###############################################

# GAME INITTIALIZATION ########################
env = Environment(render=render, pendulum_length=0.12, stacking_steps=1) 
print('state space:', env.state_space_size)
print('action space:', env.action_space_size)
###############################################

# PLOT INITIALIZATION #########################
plt.ion()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.grid(True)
plt.show()
###############################################

# AGENT INIT ####################################
input_shape = len(env.state_space_size)
print('Network Input Shape:', input_shape)
agent = Agent(input_shape=input_shape, 
            num_actions=env.action_space_size,
            learning_rate=learning_rate,
            max_epsilon=1.0, 
            min_epsilon=0.05, 
            epsilon_decay_steps=1000, 
            epsilon_decay=True) 
#################################################

# LOAD A PRETRAINED MODEL #######################
# uncomment the following to load a trained model
# trained_model_name = 'trainings/16x32_5000_140_1592020262.model'
# last_episode = 5000
# model = tf.keras.models.load_model(trained_model_name)
# print(model.summary())
# print('Starting from a trained model')
# agent.main_model = model
# agent.target_model = model
# agent.max_epsilon = 0.1
#################################################

episode = last_episode
episode_reward_list = []

### MAIN LOOP ###################################
for episode in range(last_episode, max_episodes):
    # switch between real time and free running
    real_time = False
    # if episode % render_real_time_every == 0:
    #     real_time = True

    # reset the environment and read the state
    _, _ = env.reset(real_time)  
    _, current_state, _, _ = env.step(1)
    current_state = np.reshape(current_state, [1, input_shape]) # reshape to make it compatible with keras 

    # initialization
    step = 0
    episode_reward = 0 
    reward = 0
    last_step = False

    ### LOOP FOR EACH EPISODE ###################
    while True:
        step += 1
        # print('step:{}'.format(step), end='\r')
        if step > max_steps:
            break
        elif step == max_steps:
            last_step = True

        # choose an action
        if np.random.random() > agent.epsilon:
            action = np.argmax(agent.get_q(current_state))
            # print('Agent  Action:{}'.format(action), end='\r')

        else:
            action = env.sample()
            # print('Random Action:{}'.format(action), end='\r')

        # take an action and read the consequent state
        _, new_state, _, reward = env.step(action)
        new_state = np.reshape(new_state, [1, input_shape])

        # train the ai agent
        episode_reward += reward 
        agent.update_replay_memory((current_state, action, reward, new_state))
        agent.train(last_step)

        # update the current state
        current_state = new_state

    # add the cuurent episode reward to plot
    episode_reward_list.append(episode_reward)

    # update epsilon
    agent.update_epsilon()

    # tensorboard stat update
    if not episode % aggregate_stats_every or episode == 1:
        average_reward = sum(episode_reward_list[-aggregate_stats_every:])/len(episode_reward_list[-aggregate_stats_every:])
        min_reward = min(episode_reward_list[-aggregate_stats_every:])
        max_reward = max(episode_reward_list[-aggregate_stats_every:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, 
        #                                 reward_min=min_reward, 
        #                                 reward_max=max_reward, 
        #                                 epsilon=agent.epsilon)
        
        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= min_reward_save:
            agent.main_model.save('models/{}_{}_{}max_{}avg_{}.model'.format(model_name,episode,max_reward,average_reward,int(time.time())))   

    # save the model every 'save_every' episodes
    if not episode % save_every:
        agent.main_model.save('models/{}_{}_{}_{}.model'.format(model_name,episode,episode_reward,int(time.time())))
    
    logging.info('espisode {} finished with score {}. current epsilon:{}'.format(episode, episode_reward, agent.epsilon))
    if episode >= max_episodes:
        logging.info('{} episode completed'.format(episode))
        exit(0) 

plt.plot(episode_reward_list, 'b-')
plt.draw()
plt.pause(0.1)
plt.savefig('reward_episode_{}.png'.format(episode))