import sys
import os
import time
import random
import numpy as np 
vortex_folder = r'C:\CM Labs\Vortex Studio 2020a\bin'
sys.path.append(vortex_folder)
import VxSim
import vxatp
import math

class Environment:
    def __init__(self):
        self.config_file = r'E:\VTSP_Construction\content\resources\config\quality_tests_with_graphics.vxc'
        self.content_file = 'Mechanism.vxmechanism'
        self.episode = 0
        self.motor_step = 0.0314159*10    # bipolar stepper motor step size is 1.8 deg 
        self.PI = 3.14159265359
        self.application = vxatp.VxATPConfig.createApplication(self, 'Inverted Pendulum', self.config_file)
        self.application.setSyncMode(VxSim.kSyncNone)   # set free running
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)

        # discretized space size
        self.num_discrete = 60
        self.num_states = 3

        # set up state limits
        if self.num_states == 4:
            self.limits_low = np.array([-self.PI, -0.6*self.PI, -2*self.PI, -10*self.PI])
            self.limits_high = np.array([self.PI, 0.6*self.PI, 2*self.PI, 10*self.PI])
        if self.num_states == 3:
            self.limits_low = np.array([-6*self.PI, -2*self.PI, -10*self.PI])
            self.limits_high = np.array([6*self.PI, 2*self.PI, 10*self.PI])            
        elif self.num_states == 2:
            self.limits_low = np.array([-2*self.PI, -10*self.PI])
            self.limits_high = np.array([2*self.PI, 10*self.PI])

        # action and state space size
        self.action_space_size = 3
        self.state_space_size = [self.num_discrete] * self.num_states
        self.encoded_state_size = self.num_discrete ** self.num_states

        # calculate discretization window
        self.disc_state_win_size = [0] * self.num_states
        for i in range(self.num_states):
            self.disc_state_win_size[i] = (self.limits_high[i] - self.limits_low[i])/self.state_space_size[i]

    ''' 
    reset the environment at the start of each episode
    '''
    def reset(self, real_time):
        self.mechanism = None
        self.interface = None
        self.motor_angle_current = 0    # this comes from the motor constraint  
        self.motor_angle_command = 0    # motor command, might be different from the actual motor angle

        # go to edit mode
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)

        # render realtime if requested otherwise free running
        if real_time:
            self.application.setSyncMode(VxSim.kSyncSoftwareAndVSync)
        else:
            self.application.setSyncMode(VxSim.kSyncNone)

        # unload the mechanism object if it exists
        if self.obj:
            self.application.getSimulationFileManager().unloadObject(self.obj)   
            self.obj = None    

        # recreate the mechanism object from the content file
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)
        
        # reassign the mechanism variable
        self.mechanism = VxSim.MechanismInterface(self.obj)
        self.interface = self.mechanism.findExtensionByName('Agent Interface')

        # reset the motor
        self.interface.getExtension().getInput('Motor Position').value = 0       

        # run 
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

        self.application.update()

        # update hud
        self.episode += 1
        self.interface.getExtension().getInput('HUD Text').value = 'Episode ' + str(self.episode)

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
        # decode the action
        # rotate motor one step left
        if action == 0:
            self.motor_angle_command -= self.motor_step
        
        # rotate motor one step right
        elif action == 1:
            self.motor_angle_command += self.motor_step
        
        # stop the motor
        else:
            pass

        # self.motor_angle_command = min(max(self.motor_angle_command, -self.PI), self.PI)

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

        # discretize and encode the state ########################################
        # discretize the state
        disc_state = self.discretize_state(state)

        # encode the state
        enc_state = self.encode_state(disc_state)
        ##########################################################################

        # reward assignment ######################################################
        success_thresh = 10 * 0.0174533  # 10 deg
        horizonal_threshold = 90 * 0.0174533  # 90 deg
        if pendulum_angle >= 0:
            diff = -1*abs(-self.PI + pendulum_angle)
        else:
            diff = -1*abs(self.PI + pendulum_angle)

        # give positive reward if the pendulum is upward    
        if -diff <= horizonal_threshold:
            self.reward = 1
        #     self.reward = (success_thresh + diff) * 5

        # # give positive reward if the pendulum is horizontal   
        # elif -diff <= horizonal_threshold:
        #     self.reward = (horizonal_threshold + diff) * 1       

        # # give negative reward if non of the above conditions are met
        # else:
        #     self.reward = (diff)    
        #########################################################################
    

        return disc_state, enc_state, self.reward

    '''
    gets a continous state and returns a discretized state
    '''
    def discretize_state(self, state):
        discrete_state = (state - self.limits_low)/self.disc_state_win_size 
        return list(discrete_state.astype(np.int8))  

    '''
    gets a continous state and returns a encoded state
    '''
    def encode_state(self, state):
        if len(state) == 2:
            enc_state = state[0] * self.num_discrete + state[1]
        elif len(state) == 3:
            enc_state = state[0] * (self.num_discrete ** 2) + state[1] * self.num_discrete + state[2]
        elif len(state) == 4:
            enc_state = state[0] * (self.num_discrete ** 3) + state[1] * (self.num_discrete ** 2) + state[2] * self.num_discrete + state[3]

        return enc_state

    '''
    returns a random action
    0: rotate one step left
    1: rotate one step right
    2: stop the motor
    '''
    def sample(self):
        return random.choice([0, 1, 2])