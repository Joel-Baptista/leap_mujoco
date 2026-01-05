import mujoco as mj
from mujoco import viewer
import numpy as np
import time

DEG2RAD = np.pi / 180.0

def main():
    
    model = mj.MjModel.from_xml_path("model/leap_grasp_ball.xml")
    data = mj.MjData(model)

    mj.mj_step(model, data)

    # action = [30, 60, -30, -60, 80, -45, -80, 45]
    # action = np.array(action) * DEG2RAD

    # data.ctrl[:] = action
    mj.mj_step(model, data)

    with viewer.launch_passive(model, data) as v:
        while True:
            
            # x, rb_trace = minimize.least_squares(x0, optimize_torque)
            print(data.sensordata, " -> ", len(data.sensordata))
            action = [30, 60, -30, -60, 80, -45, -80, 45, 0, 0, 0, 0]
            action = np.array(action) * DEG2RAD
            
            # data.ctrl[:] = action
            mj.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep) 
            
                
    
            
if __name__ == "__main__":
    main()
