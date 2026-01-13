import threading
from pynput import keyboard
import time
import numpy as np
import copy

import struct
import os

from interfaces.base_interface import BaseInterface


class KeyboardInterface(BaseInterface):
    def __init__(self, model, data, scale = [0.01, 0.01]) -> None:
        self.last_press_time = time.time()
        super().__init__(model, data, scale)

    def input_to_robot_action(self, device_input):

        robot_action = np.zeros(22)

        if device_input == 'w':
            print('Move forward')
            robot_action[2] = 1
        elif device_input == 's':
            robot_action[2] = -1
            print('Move backward')
        elif device_input == 'a':
            robot_action[0] = 1
            print('Move left')
        elif device_input == 'd':
            robot_action[0] = -1
            print('Move right')
        elif device_input == 'i':
            robot_action[7] = -1
            print('Wrist up')
        elif device_input == 'k':
            robot_action[7] = 1
            print('Wrist down')
        elif device_input == 'j':
            robot_action[5] = -1
            print('roll left')
        elif device_input == 'l':
            robot_action[5] = 1
            print('roll left')
        elif device_input == '1':
            robot_action[9] = 1
            robot_action[10] = 1
            robot_action[11] = 1
            print('flexion forefinger')
        elif device_input == '!':
            robot_action[9] = -1
            robot_action[10] = -1
            robot_action[11] = -1
            print('Close forefinger')
        elif device_input == '2':
            robot_action[13] = 1
            robot_action[14] = 1
            robot_action[15] = 1
            print('Flexion middle finger')
        elif device_input == '"':
            robot_action[13] = -1
            robot_action[14] = -1
            robot_action[15] = -1
            print('Close middle finger')
        elif device_input == '3':
            robot_action[17] = 1
            robot_action[18] = 1
            robot_action[19] = 1
            print('Flexion ring finger')
        elif device_input == '#':
            robot_action[17] = -1
            robot_action[18] = -1
            robot_action[19] = -1
            print('Close ring finger')
        elif device_input == '4':
            robot_action[22] = 1
            robot_action[23] = 1
            robot_action[24] = 1
            print('Flexion little finger')
        elif device_input == '$':
            robot_action[22] = -1
            robot_action[23] = -1
            robot_action[24] = -1
            print('Close little finger')
        elif device_input == '5':
            robot_action[25] = -1
            robot_action[26] = 1
            robot_action[27] = -1
            robot_action[28] = -1
            robot_action[29] = -1
            print('Flexion thumb finger')
        elif device_input == '%':
            robot_action[25] = 1
            print('Close thumb finger')
        elif device_input == '6':
            robot_action[9] = 1
            robot_action[10] = 1
            robot_action[11] = 1
            robot_action[13] = 1
            robot_action[14] = 1
            robot_action[15] = 1
            robot_action[17] = 1
            robot_action[18] = 1
            robot_action[19] = 1
            robot_action[22] = 1
            robot_action[23] = 1
            robot_action[24] = 1
            robot_action[25] = 1
            robot_action[26] = -1
            robot_action[27] = 1
            robot_action[28] = 1
            robot_action[29] = 1
            print('Flexion all fingers')
        elif device_input == '&':
            robot_action[9] = -1
            robot_action[10] = -1
            robot_action[11] = -1
            robot_action[13] = -1
            robot_action[14] = -1
            robot_action[15] = -1
            robot_action[17] = -1
            robot_action[18] = -1
            robot_action[19] = -1
            robot_action[22] = -1
            robot_action[23] = -1
            robot_action[24] = -1
            robot_action[25] = -1
            robot_action[26] = 1
            robot_action[27] = -1
            robot_action[28] = -1
            robot_action[29] = -1
            print('close all fingers')
        elif device_input == 'q':
            print('Quit')
            return False  # Stop listener
        elif device_input == "space":
            pass
        elif device_input == "shift":
            pass
        elif device_input == "up":
            pass
        elif device_input == "down":
            pass
        elif device_input == "left":
            pass
        elif device_input == "right":
            pass
        
        return robot_action

    # Function to read from the joystick device
    def read_input(self):
        def on_press(key):
            print(key)
            try:
                self.last_press_time = time.time()
                self.current_readings = key.char
            except AttributeError:
                if key == keyboard.Key.space:
                    self.current_readings = "space"
                elif key == keyboard.Key.shift:
                    self.current_readings = "shift"
                elif key == keyboard.Key.up:
                    self.current_readings = "up"
                elif key == keyboard.Key.down:
                    self.current_readings = "down"
                elif key == keyboard.Key.left:
                    self.current_readings = "left"
                elif key == keyboard.Key.right:
                    self.current_readings = "right"
                    
        # Set up the listener
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


