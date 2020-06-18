import numpy as np
import random
from collections import deque
import time
import random
from .cnn import CNN 
# from .modified_tensor_board import  ModifiedTensorBoard

REPLAY_MEMORY_SIZE = 1000000
MIN_REPLAY_MEMORY_SIZE = 2000
MINIBATCH_SIZE = 64
MODEL_NAME = '24x24'

class Agent:
    def __init__(self, input_shape, num_actions, epsilon_decay=False, epsilon=0.01,
                max_epsilon=1.0, min_epsilon=0.1, epsilon_decay_steps=100000, 
                gamma=0.99,learning_rate=0.01):
        self.input_shape = input_shape
        self.num_actions = num_actions

        # additional arguments 
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma
        self.learning_rate = learning_rate

        # select epsilon 
        if self.epsilon_decay:
            self.epsilon = self.max_epsilon
        else:
            self.epsilon = epsilon
        
        # main model, this is what we use for fitting, it gets trained every step
        self.main_model = CNN(input_shape, num_actions, self.learning_rate).model

        # replay memory 
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # tensorboard initialization, it is the modified version of stock tensorboard
        # to reduce the frequency of log file update
        # self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}.h5'.format(MODEL_NAME, int(time.time())))

        # this is used to update epsilon
        self.step = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_q(self, state):
        q = self.main_model.predict(state)[0]    # [0] is used to get the first and only member of a list of list
        return q

    def train(self, last_step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.zeros((MINIBATCH_SIZE, self.input_shape), dtype=np.int16)
        new_states = np.zeros((MINIBATCH_SIZE, self.input_shape), dtype=np.int16)
        actions = np.zeros(MINIBATCH_SIZE, dtype=np.int8)
        rewards = np.zeros(MINIBATCH_SIZE)        
        for i in range(MINIBATCH_SIZE):
            current_states[i] = minibatch[i][0]
            actions[i] = minibatch[i][1]
            rewards[i] = minibatch[i][2]            
            new_states[i] = minibatch[i][3]

        target = self.main_model.predict(current_states)
        target_next = self.main_model.predict(new_states)

        for i in range(MINIBATCH_SIZE):
            target[i][actions[i]] = rewards[i] + self.gamma * (np.amax(target_next[i]))

        self.main_model.fit(current_states, target, batch_size=MINIBATCH_SIZE, verbose=0)
                            

    def update_epsilon(self):
        if self.epsilon_decay:
            self.epsilon = self.max_epsilon - self.step*((self.max_epsilon-self.min_epsilon)/self.epsilon_decay_steps)
            self.epsilon = max(self.min_epsilon, self.epsilon)      
            self.step += 1      
