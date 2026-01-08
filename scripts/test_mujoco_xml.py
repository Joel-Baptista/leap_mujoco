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
            print(data.contact)

            for contact in data.contact:
                print(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom1))
                print(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom2))
                print("----------------------------------")
            
            # data.ctrl[:] = action
            mj.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep) 
            
                
    
            
if __name__ == "__main__":
    main()
