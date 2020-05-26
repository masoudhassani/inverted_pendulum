import sys
import os
import time
import random
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
        self.motor_angle_current = 0    # this comes from the motor constraint  
        self.motor_angle_command = 0    # motor command, might be different from the actual motor angle
        self.PI = 3.14159265359
        self.application = vxatp.VxATPConfig.createApplication(self, 'Inverted Pendulum', self.config_file)
        self.application.setSyncMode(VxSim.kSyncNone)   # set free running
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)

        # limits of state variables, motor angle, speed, pendulum angle, speed
        self.limits = [[-self.PI, self.PI],
                       [-0.6*self.PI, 0.6*self.PI],
                       [-2*self.PI, 2*self.PI],
                       [-4*self.PI, 4*self.PI]]

        # discretized space size
        num_disc_states = 15
        self.disc_state_size = [(i[1] - i[0])/num_disc_states for i in self.limits]

        # space/action size
        self.state_space_size = [num_disc_states]*4
        self.action_space_size = 2

    def reset(self):
        # self.application = None
        self.mechanism = None
        self.interface = None
        self.application.getSimulationFileManager().unloadObject(self.content_file)
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)

        obj = self.application.getSimulationFileManager().loadObject(self.content_file)
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

        self.mechanism = VxSim.MechanismInterface(obj)
        self.interface = self.mechanism.findExtensionByName('Agent Interface')
        self.application.update()
        obj = None

        # update hud
        self.episode += 1
        self.interface.getExtension().getInput('HUD Text').value = 'Episode ' + str(self.episode)

        # reset step reward
        self.reward = 0

        # return the current state
        return [0, 0]

   # step the simulation and return the state
    def step(self, action):
        # decode the action
        # rotate motor one step left
        if action == 0:
            self.motor_angle_command -= self.motor_step
        
        # rotate motor one step right
        else:
            self.motor_angle_command += self.motor_step

        # push the motor command to the constraint
        self.interface.getExtension().getInput('Motor Position').value = self.motor_angle_command

        # start the simulation
        self.application.update()

        # read current state
        motor_angle = self.interface.getExtension().getOutput('Motor Angle').getValue()
        motor_velocity = self.interface.getExtension().getOutput('Motor Angular Velocity').getValue()        
        pendulum_angle = self.interface.getExtension().getOutput('Pendulum Angle').getValue()
        pendulum_velocity = self.interface.getExtension().getOutput('Pendulum Angular Velocity').getValue()
        if pendulum_angle > 2*self.PI:
            pendulum_angle = pendulum_angle - int(pendulum_angle/2*self.PI)*2*self.PI
        elif pendulum_angle < -2*self.PI:
            pendulum_angle = pendulum_angle + int(-pendulum_angle/2*self.PI)*2*self.PI

        state = [motor_angle, motor_velocity, pendulum_angle, pendulum_velocity]
        disc_state = self.discretize_state(state)

        return disc_state, self.reward

    # gets a continous state and returns a discretized state
    def discretize_state(self, state):
        disc_state = []
        for i in range(len(state)):
            disc_state.append(int((state[i] - self.limits[i][0])/self.disc_state_size[i]))
        
        return disc_state


    # returns a random action
    # 0: rotate one step left
    # 1: rotate one step right
    def sample(self):
        return random.choice([0, 1])