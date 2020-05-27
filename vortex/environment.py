import sys
import os
import time
import random
import numpy as np 
vortex_folder = r'C:\CM Labs\Vortex Studio 2020a\bin'
sys.path.append(vortex_folder)
import VxSim
import vxatp

class Environment:
    def __init__(self):
        self.config_file = r'E:\VTSP_Construction\content\resources\config\quality_tests_with_graphics.vxc'
        self.content_file = 'Mechanism.vxmechanism'
        self.episode = 0
        self.motor_step = 0.0314159    # bipolar stepper motor step size is 1.8 deg 
        self.PI = 3.14159265359
        self.application = vxatp.VxATPConfig.createApplication(self, 'Inverted Pendulum', self.config_file)
        self.application.setSyncMode(VxSim.kSyncNone)   # set free running
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)

        # discretized space size
        NUM_DISCRETE = 30
        self.num_states = 2

        # set up state limits
        if self.num_states == 4:
            self.limits_low = np.array([-self.PI, -0.6*self.PI, -2*self.PI, -8*self.PI])
            self.limits_high = np.array([self.PI, 0.6*self.PI, 2*self.PI, 8*self.PI])
        elif self.num_states == 2:
            self.limits_low = np.array([-2*self.PI, -8*self.PI])
            self.limits_high = np.array([2*self.PI, 8*self.PI])

        
        # calculate discretization window
        self.state_space_size = [NUM_DISCRETE] * self.num_states
        self.disc_state_win_size = [0] * self.num_states
        for i in range(self.num_states):
            self.disc_state_win_size[i] = (self.limits_high[i] - self.limits_low[i])/self.state_space_size[i]

        # action size
        self.action_space_size = 2

    ''' 
    reset the environment at the start of each episode
    '''
    def reset(self):
        self.mechanism = None
        self.interface = None
        self.motor_angle_current = 0    # this comes from the motor constraint  
        self.motor_angle_command = 0    # motor command, might be different from the actual motor angle

        # unload the mechanism object if it exists
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)
        if self.obj:
            self.application.getSimulationFileManager().unloadObject(self.obj)       

        # recreate the mechanism object from the content file
        self.obj = self.application.getSimulationFileManager().loadObject(self.content_file)
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

        # reassign the mechanism variable
        self.mechanism = VxSim.MechanismInterface(self.obj)
        self.interface = self.mechanism.findExtensionByName('Agent Interface')
        self.application.update()

        # update hud
        self.episode += 1
        self.interface.getExtension().getInput('HUD Text').value = 'Episode ' + str(self.episode)

        # reset step reward
        self.reward = 0

        print('--------------- environment was successfully reset --------------------')
        print('--------------------- starting a new episode--------------------')

        # return the current state
        return [0] * self.num_states

    '''
    step the simulation and return the state
    '''
    def step(self, action):
        done = False
        self.reward = 0
        # decode the action
        # rotate motor one step left
        if action == 0:
            self.motor_angle_command -= self.motor_step
        
        # rotate motor one step right
        else:
            self.motor_angle_command += self.motor_step
        self.motor_angle_command = min(max(self.motor_angle_command, -self.PI), self.PI)

        # push the motor command to the constraint
        self.interface.getExtension().getInput('Motor Position').value = self.motor_angle_command

        # start the simulation
        self.application.update()

        # read current state
        motor_angle = self.interface.getExtension().getOutput('Motor Angle').getValue()
        motor_velocity = self.interface.getExtension().getOutput('Motor Angular Velocity').getValue()        
        pendulum_angle = self.interface.getExtension().getOutput('Pendulum Angle').getValue()
        pendulum_velocity = self.interface.getExtension().getOutput('Pendulum Angular Velocity').getValue()   

        if self.num_states == 4:
            state = np.array([motor_angle, motor_velocity, pendulum_angle, pendulum_velocity])

        elif self.num_states == 2:     
            state = np.array([pendulum_angle, pendulum_velocity])

        if pendulum_angle > 2*self.PI:
            pendulum_angle = pendulum_angle - int(pendulum_angle/2*self.PI)*2*self.PI
        elif pendulum_angle < -2*self.PI:
            pendulum_angle = pendulum_angle + int(-pendulum_angle/2*self.PI)*2*self.PI

        # discretize the state
        disc_state = self.discretize_state(state)

        # reward assignment
        self.reward = -self.PI + abs(pendulum_angle)

        return disc_state, self.reward

    '''
    gets a continous state and returns a discretized state
    '''
    def discretize_state(self, state):
        discrete_state = (state - self.limits_low)/self.disc_state_win_size 
        return list(discrete_state.astype(np.int8))  

    '''
    returns a random action
    0: rotate one step left
    1: rotate one step right
    '''
    def sample(self):
        return random.choice([0, 1])