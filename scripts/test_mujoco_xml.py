import mujoco as mj
from mujoco import viewer
import numpy as np
import time

from interfaces.joystick_interface import JoyStickInterface
from interfaces.keyboard_interface import KeyboardInterface

DEG2RAD = np.pi / 180.0

def main():
    
    model = mj.MjModel.from_xml_path("model/leap_grasp_ball.xml")
    data = mj.MjData(model)

    mj.mj_step(model, data)


    try:
        interface = JoyStickInterface(model, data)
    except:
        interface = KeyboardInterface(model, data)

    with viewer.launch_passive(model, data) as v:
        while True:
            # print("In sim loop:", data.ctrl.ravel().copy())
            reset = interface.update_robot_state()

            if reset:
                mj.mj_resetData(model, data)

            # data.ctrl[:] = action
            mj.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep) 
            
                
    
            
if __name__ == "__main__":
    main()
