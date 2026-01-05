from envs.leap_grasp import LeapGrasp 
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit

def main():
    
    def make_env():
        env = LeapGrasp()
        return TimeLimit(env, max_episode_steps=1000)

    env = SubprocVecEnv([make_env for _ in range(1)])
    eval_env = TimeLimit(LeapGrasp(), max_episode_steps=200)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/SAC",
                                  log_path="./logs/SAC", eval_freq=10_000,
                                  deterministic=True, render=False, n_eval_episodes=20)

    agent = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, 
        batch_size=64,
        buffer_size=1000000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        train_freq=(4, "step"),
        gradient_steps=1,
        learning_starts=10000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, tensorboard_log="./tensorboard_logs/"
    )
    
    print(agent.policy)
    agent.learn(total_timesteps=1_000_000, callback=eval_callback)
    
    test_env = LeapGrasp()
    obs = test_env.reset()
    
    for _ in range(200):
        action, _ = agent.predict(env.observation_space.sample())

        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        time.sleep(0.01)
        
        if terminated or truncated:
            obs = test_env.reset()
        
    env.close()
    test_env.close()
if __name__ == "__main__":
    main()