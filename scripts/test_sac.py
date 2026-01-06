from envs.leap_grasp import LeapGrasp
import numpy as np
import time

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit

HEALTHY_RANGE = (0.07, 0.2)
SKIP_FRAMES = 10
RESET_NOISE = 0.0
CTRL_COST = 0.05


def main():

    env = TimeLimit(
        LeapGrasp(render_mode="human"),
        max_episode_steps=200,
    )

    agent = SAC("MlpPolicy", env, verbose=1)

    mean_weight = np.mean(
        [param.data.numpy().mean() for param in agent.policy.parameters()]
    )
    print("Mean of the weights in the policy:", mean_weight)

    agent = agent.load("logs/SAC/best_model.zip", env=env)

    mean_weight = np.mean(
        [param.data.numpy().mean() for param in agent.policy.parameters()]
    )
    print("Mean of the weights in the policy:", mean_weight)

    obs, info = env.reset()
    
    cummulative_reward = 0

    for e in range(0, 5): 
        for _ in range(200):
            action, _ = agent.predict(obs, deterministic=True)

            # print("Action:", action)

            obs, reward, terminated, truncated, info = env.step(action)
            cummulative_reward += reward
            # print("Observation:", obs[2])
            # print(info)
            env.render()
            # time.sleep(0.001)

            if terminated or truncated:
                print("--------------------------------------------------------")
                obs, info = env.reset()

    env.close()

    print("Episode Reward: ", cummulative_reward / e)

if __name__ == "__main__":
    main()
