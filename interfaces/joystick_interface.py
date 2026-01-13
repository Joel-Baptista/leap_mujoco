import time
import numpy as np
import copy

import struct
import os

from interfaces.base_interface import BaseInterface

from scipy.spatial.transform import Rotation as R


class JoyStickInterface(BaseInterface):
    def __init__(self, model, data, scale = [0.01, 0.01]) -> None:
        self.device_path = self.find_joystick_device()
        print(f"Using joystick device: {self.device_path}")

        super().__init__(model, data, scale)


    # Function to read from the joystick device
    def read_input(self):
        with open(self.device_path, 'rb') as js:
            while True:
                # Read 8 bytes from the joystick device
                evbuf = js.read(8)
                if evbuf:
                    # Unpack the event buffer
                    self.last_press_time = time.time()
                    _, value, event_type, number = struct.unpack('IhBB', evbuf)
                    # Process the event
                    if event_type & 0x01:
                        if not (f"Axis{number}" in self.current_readings):
                            self.current_readings[f"Button{number}"] = 0                             
                        self.current_readings[f"Button{number}"] = value

                    elif event_type & 0x02:
                        if not (f"Axis{number}" in self.current_readings):
                            self.current_readings[f"Axis{number}"] = 0
                        self.current_readings[f"Axis{number}"] = value / self.analog_scale
            
    # Find the joystick device  
    def find_joystick_device(self):
        for fn in os.listdir('/dev/input'):
            if fn.startswith('js'):
                return os.path.join('/dev/input', fn)
        raise Exception("No joystick device found")

    def input_to_robot_action(self, input_action):
        # Button0 -> X
        # Button1 -> O
        # Button2 -> Triangle
        # Button3 -> Square
        # Button4 -> L1
        # Button5 -> R1
        # Button6 -> L2
        # Button7 -> R2
        # Button8 -> Select
        # Button9 -> Start
        # Button10 -> PS
        # Button11 -> L3
        # Button12 -> R3
        # Button13 -> Up
        # Button14 -> Down
        # Button15 -> Left
        # Button16 -> Right
        # Axis0 -> Left stick left/right
        # Axis1 -> Left stick up/down
        # Axis2 -> L2
        # Axis3 -> Right stick left/right
        # Axis4 -> Right stick up/down
        # Axis5 -> R2

        robot_action = np.zeros(22)
        if len(input_action.keys()) == 0:
            return np.array([0] * 22)
        
        robot_action[0] = -input_action["Axis0"] # Arm X
        robot_action[1] = -input_action["Axis1"] # Arm Y
        robot_action[2] = input_action["Axis2"] - input_action["Axis5"]  # Arm Z
        robot_action[3] =  - input_action["Axis3"] * (1 - input_action["Button5"])  # Arm angular left/right
        robot_action[4] =  input_action["Axis4"] * (1 - input_action["Button5"]) # Arm angular up/down
        robot_action[5] =  input_action["Axis3"] * input_action["Button5"] # Full rotation
    
        robot_action[6] = max(input_action["Button13"], input_action["Button0"]) * (1 - input_action["Button4"] * 2)   # Forefinger
        robot_action[8] = max(input_action["Button13"], input_action["Button0"]) * (1 - input_action["Button4"] * 2)   # Forefinger
        robot_action[9] = max(input_action["Button13"], input_action["Button0"]) * (1 - input_action["Button4"] * 2)   # Forefinger
        
        robot_action[10] = max(input_action["Button15"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Middle finger
        robot_action[12] = max(input_action["Button15"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Middle finger
        robot_action[13] = max(input_action["Button15"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Middle finger
        
        robot_action[14] = max(input_action["Button14"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Ring finger
        robot_action[16] = max(input_action["Button14"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Ring finger
        robot_action[17] = max(input_action["Button14"], input_action["Button0"]) * (1 - input_action["Button4"] * 2) # Ring finger
        
        robot_action[18] = max(input_action["Button13"], input_action["Button2"]) * (1 - input_action["Button4"] * 2) # Thumb
        robot_action[19] = max(input_action["Button16"], input_action["Button3"]) * (1 - input_action["Button4"] * 2) 
        robot_action[20] = max(input_action["Button13"], input_action["Button0"]) * (1 - input_action["Button4"] * 2)
        robot_action[21] = max(input_action["Button13"], input_action["Button0"]) * (1 - input_action["Button4"] * 2)

        if input_action["Button8"]:
            self.reset_env = True
        
        return robot_action
