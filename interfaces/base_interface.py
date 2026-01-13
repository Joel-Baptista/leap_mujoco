import mujoco as mj

import threading
from pynput import keyboard
import time
import numpy as np
import copy

import struct
import os

from scipy.spatial.transform import Rotation as R

class BaseInterface:
    def __init__(self, model, data, scale = [0.1, 0.1]) -> None:
        self.model = model
        self.data = data

        self.act_mean = np.mean(self.model.jnt_range[:22, :], axis=1)
        self.act_rng = 0.5 * (
            self.model.jnt_range[:22, 1] - self.model.jnt_range[:22, 0]
        )

        self.analog_scale = 32767
        self.current_readings = {}
        self.reset_env = False

        self.init_ctrl = np.array([0.0] * 22)
        self.init_ctrl[2] = 0.3
        self.init_ctrl[19] = 1.57
        self.last_ctrl = self.init_ctrl

        self.scale = np.array([scale[0]] * 6 + [scale[1]] * 16)
        # Create and start the keyboard listener thread
        keyboard_listener_thread = threading.Thread(target=self.read_input)
        keyboard_listener_thread.daemon = True  # This ensures the thread exits when the main program exits
        keyboard_listener_thread.start()
        

    def update_robot_state(self):
        
        last_action = (self.last_ctrl - self.act_mean) / self.act_rng
        action = self.input_to_robot_action(copy.deepcopy(self.current_readings))

        actuated = np.abs(action) > 0.01

        if any(actuated):
            last_action[actuated] += action[actuated] * self.scale[actuated]

        last_action = np.clip(last_action, -1.0, 1.0)
        ctrl = self.act_mean + last_action * self.act_rng

        ctrl = np.clip(
            ctrl,
            self.model.jnt_range[:22, 0],
            self.model.jnt_range[:22, 1],
        )

        self.data.ctrl = ctrl[6:]
        self.last_ctrl = ctrl

        self.data.mocap_pos[1] = ctrl[:3]
        self.data.mocap_quat[1] = R.from_euler("xyz", ctrl[3:6]).as_quat(scalar_first=True)

        if self.reset_env:
            self.reset_env = False
            self.last_ctrl = self.init_ctrl
            return True

        return False

    # Function to read from the joystick device
    def read_input(self):
        raise NotImplementedError("Implement reading inputs from device")

    def input_to_robot_action(self, device_input):
        raise NotImplementedError("Implement map between device input and robot input")

