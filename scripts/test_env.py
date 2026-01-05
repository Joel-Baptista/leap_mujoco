from gymnasium.envs.registration import register
import gymnasium as gym

import numpy as np

from envs.leap_grasp import LeapGrasp
import time


if __name__ == "__main__":

    register(
        id="LeapGrasp-v0",
        entry_point="test_env:LeapGrasp",
    )

    env = gym.make(
        "LeapGrasp-v0",
        render_mode="human",
        max_translation=0.1,
        max_rotation=0.1,
        )
    
    mean_stat = []
    for i in range(50):

        env.reset()

        ep_count = 0
        st = time.time()
        t = 10000000
        num_contacts_ep = []
        timestep = 0
        first = True
        while True:
            timestep += 1

            action = np.zeros(env.action_space.shape)
            action[19] = -1.57
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated or time.time() - st > t or timestep > 200:
                env.reset()
                st = time.time()

                ep_count += 1
                timestep = 0
                first = True
                if ep_count > 50:
                    break

        mean_stat.append(np.mean(num_contacts_ep))
