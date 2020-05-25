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


    def reset(self):
        self.application = vxatp.VxATPConfig.createApplication(self, 'Inverted Pendulum', self.config_file)
        self.application.setSyncMode(VxSim.kSyncNone)   # set free running
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
        pendulum_angle = self.interface.getExtension().getOutput('Pendulum Angle').getValue()
        pendulum_velocity = self.interface.getExtension().getOutput('Pendulum Angular Velocity').getValue()
        state = [pendulum_angle, pendulum_velocity]

        return state, self.reward

    # returns a random action
    # 0: rotate one step left
    # 1: rotate one step right
    def sample(self):
        return random.choice([0, 1])