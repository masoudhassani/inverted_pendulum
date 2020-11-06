import sys
import os
import time
import random
import numpy as np 
vortex_folder = r'C:\CM Labs\Vortex Studio 2020b\bin'
sys.path.append(vortex_folder)
import Vortex
import vxatp3
import math


class Environment:
    def __init__(self, render=True, pendulum_length=0.12, stacking_steps=10):
        if render:
            self.config_file = r'resources\setup.vxc'
        else:
            self.config_file = r'resources\setup_no_3d.vxc'

        self.render = render
        self.stacking_steps = stacking_steps
        self.length = pendulum_length
        self.content_file = 'Mechanism.vxmechanism'
        self.episode = 0
        self.PI = 3.14159265359
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Inverted Pendulum', self.config_file)
        self.application.setSyncMode(Vortex.kSyncNone)   # set free running
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)

        # main parameters #######################################
        self.num_discrete_high = 1000
        self.num_discrete_mid = 600
        self.num_discrete_low = 50
        self.num_states = 2
        #########################################################

        # motor settings ########################################
        # self.motor_step_size = 0.0314159/8    # bipolar stepper motor step size is 1.8 deg, we set the driver to 1/8 micro steps -> 0.003926
        # self.motor_step_max = 40   # 0.03766 rad/s or 9 microsteps
        # self.motor_step_min = 40   # 0.02026 rad/s or 5 micro steps
        self.motor_step_size = 0.0314159    # bipolar stepper motor step size is 1.8 deg
        self.motor_step_max = 5   
        self.motor_step_min = 5   

        # construct a list containing possible motor step sizes
        self.motor_step_factor = [i for i in range(-self.motor_step_max, -self.motor_step_min+1)]   # reverse
        self.motor_step_factor.append(0)                                                            # stop
        temp = [i for i in range(self.motor_step_min, self.motor_step_max+1)]                       # forward
        self.motor_step_factor += temp

        self.action_space_size = len(self.motor_step_factor)
        #########################################################

        # set up state limits ###################################
        if self.num_states == 4:
            self.state_space_size = [self.num_discrete_low, self.num_discrete_low, self.num_discrete_mid, self.num_discrete_high]
            self.limits_low = np.array([-self.PI, -4*self.PI, -2*self.PI, -6*self.PI])
            self.limits_high = np.array([self.PI, 4*self.PI, 2*self.PI, 6*self.PI])
            
        if self.num_states == 3:
            self.state_space_size = [self.num_discrete_low, self.num_discrete_mid, self.num_discrete_high]
            self.limits_low = np.array([-4*self.PI, -2*self.PI, -6*self.PI])
            self.limits_high = np.array([4*self.PI, 2*self.PI, 6*self.PI])    
   
        elif self.num_states == 2:
            self.state_space_size = [self.num_discrete_mid, self.num_discrete_high]
            self.limits_low = np.array([-2*self.PI, -6*self.PI])
            self.limits_high = np.array([2*self.PI, 6*self.PI])
        #########################################################

        # encoded state space size
        self.encoded_state_size = np.prod(self.state_space_size)

        # calculate discretization window
        self.disc_state_win_size = [0] * self.num_states
        for i in range(self.num_states):
            self.disc_state_win_size[i] = (self.limits_high[i] - self.limits_low[i])/self.state_space_size[i]

        # init action choises for random action selection
        self.action_choices = [i for i in range(self.action_space_size)]

    ''' 
    reset the environment at the start of each episode
    '''
    def reset(self, real_time):
        self.mechanism = None
        self.interface = None
        self.motor_angle_current = 0    # this comes from the motor constraint  
        self.motor_angle_command = 0    # motor command, might be different from the actual motor angle
        self.pendulum_angle_prev = 0
        self.energy_prev = 0
        self.peak_pos = 0
        self.peak_neg = 0
        self.peak_pos_set = False
        self.peak_neg_set = False
        self.upward = False
        self.downward = False

        # go to edit mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

        # render realtime if requested otherwise free running
        if real_time and self.render:
            self.application.setSyncMode(Vortex.kSyncSoftwareAndVSync)
        else:
            self.application.setSyncMode(Vortex.kSyncNone)

        # unload the mechanism object if it exists
        if self.obj:
            self.application.getSimulationFileManager().unloadObject(self.obj)   
            self.obj = None    

        # recreate the mechanism object from the content file
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)
        
        # reassign the mechanism variable
        self.mechanism = Vortex.MechanismInterface(self.obj)
        self.interface = self.mechanism.findExtensionByName('Agent Interface')

        # reset the motor
        self.interface.getExtension().getInput('Motor Position').value = 0       

        # run 
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

        self.application.update()

        # update hud
        self.episode += 1
        self.set_text('Episode ' + str(self.episode))

        # reset step reward
        self.reward = 0

        print('--------------- environment was successfully reset --------------------')
        print('--------------------- starting a new episode--------------------')

        # return the current state, discrete and encoded
        return [0] * self.num_states, 0

    '''
    step the simulation and return the state
    '''
    def step(self, action):
        self.reward = 0

        for _ in range(self.stacking_steps):
            self.motor_angle_command += self.motor_step_factor[action]*self.motor_step_size

            # push the motor command to the constraint
            self.interface.getExtension().getInput('Motor Position').value = self.motor_angle_command

            # start the simulation
            self.application.update()

            # read current state ###################################################
            motor_angle = self.interface.getExtension().getOutput('Motor Angle').getValue()
            motor_velocity = self.interface.getExtension().getOutput('Motor Angular Velocity').getValue()        
            pendulum_angle = self.interface.getExtension().getOutput('Pendulum Angle').getValue()
            pendulum_velocity = self.interface.getExtension().getOutput('Pendulum Angular Velocity').getValue()   

            if pendulum_angle > 2*self.PI:
                pendulum_angle = pendulum_angle - int(pendulum_angle/(2*self.PI))*2*self.PI
            elif pendulum_angle < -2*self.PI:
                pendulum_angle = pendulum_angle + int(-pendulum_angle/(2*self.PI))*2*self.PI

            if self.num_states == 4:
                state = np.array([motor_angle, motor_velocity, pendulum_angle, pendulum_velocity])

            elif self.num_states == 3:
                state = np.array([motor_velocity, pendulum_angle, pendulum_velocity])

            elif self.num_states == 2:     
                state = np.array([pendulum_angle, pendulum_velocity])
            ##########################################################################

        # reward assignment ######################################################  
        success_thresh = 10 * 0.0174533  # 10 deg
        horizonal_threshold = 90 * 0.0174533  # 90 deg
        if pendulum_angle >= 0:
            diff = abs(-self.PI + pendulum_angle)
        else:
            diff = abs(self.PI + pendulum_angle)  
        if diff <= success_thresh:
            self.upward = True
            self.motor_step_size = 0.0314159 /4

        else:   
            self.motor_step_size = 0.0314159     
            energy = self.pendulum_energy(pendulum_angle, pendulum_velocity)
            if energy >= 1.01*self.energy_prev:
                self.reward = 1
            else:
                self.reward = -1

            self.energy_prev = energy

            if pendulum_angle >= 0:
                if pendulum_angle < self.pendulum_angle_prev and not self.peak_pos_set:
                    self.peak_pos = self.pendulum_angle_prev
                    self.peak_pos_set = True
                    self.peak_neg_set = False
                    if self.peak_pos > self.peak_neg:
                        self.reward = 5
                    else: 
                        self.reward = -5
            
            elif pendulum_angle < 0:
                if pendulum_angle > self.pendulum_angle_prev and not self.peak_neg_set:
                    self.peak_neg = abs(self.pendulum_angle_prev)
                    self.peak_pos_set = False
                    self.peak_neg_set = True
                    if self.peak_neg > self.peak_pos:
                        self.reward = 5
                    else: 
                        self.reward = -5   

            self.pendulum_angle_prev = pendulum_angle 
        
        if self.upward:
            self.downward = False
            if diff <= success_thresh:
                self.reward = 5 + ((success_thresh-diff) * 100 / success_thresh) - abs(pendulum_velocity/self.PI) * 50
            else:
                # self.reward = -500
                self.upward = False
                self.downward = True
        #########################################################################

        # discretize and encode the state ########################################
        # discretize the state
        disc_state = self.discretize_state(state)

        # encode the state
        enc_state = self.encode_state(disc_state)
        ##########################################################################

        return state, disc_state, enc_state, self.reward

    '''
    gets a continous state and returns a discretized state
    '''
    def discretize_state(self, state):
        discrete_state = []
        for i in range(len(state)):
            discrete_state.append( int((state[i] - self.limits_low[i])/self.disc_state_win_size[i]) )
        return discrete_state
        # return list(discrete_state.astype(np.int8))  

    '''
    gets a continous state and returns a encoded state
    '''
    def encode_state(self, state):
        if len(state) == 2:
            enc_state = state[0] * self.state_space_size[1] + state[1]
        elif len(state) == 3:
            enc_state = state[0] * (self.state_space_size[1] * self.state_space_size[2]) + state[1] * self.state_space_size[2] + state[2]
        elif len(state) == 4:
            enc_state = state[0] * (self.state_space_size[1] * self.state_space_size[2] * self.state_space_size[3]
                                        ) + state[1] * (self.state_space_size[2] * self.state_space_size[3]) + state[2] * self.state_space_size[3] + state[3]

        return enc_state

    '''
    returns a random action
    0: rotate one step left
    1: rotate one step right
    2: stop the motor
    '''
    def sample(self):
        return random.choice(self.action_choices)

    '''
    calculate the current energy of pendulum per unit of mass
    '''
    def pendulum_energy(self, angle, velocity):
        l = self.length
        dh = l * (1 - math.cos(angle))
        u = 9.81 * dh
        k = 0.5 * (l*velocity)**2

        return u + k

    '''
    set hud text
    '''
    def set_text(self, text):
        self.interface.getExtension().getInput('HUD Text').value = text

    '''
    decay hyper parameters
    '''
    def parameter_decay(self, current, min_value, factor, max_value=1.0, episode=0, type='linear'):
        if type == 'linear':
            return max(min_value, current*factor)
        elif type == 'exp':
            return max(min_value, max_value*math.exp(-factor*episode))