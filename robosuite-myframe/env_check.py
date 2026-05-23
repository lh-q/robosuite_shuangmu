import sys
sys.path.insert(0, "/home/lh/robosuite")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config

from stable_baselines3.common.env_checker import check_env

#from grippers.fuelingpeg import FuelingPegGripper
if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "MyCGAssembly"

    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env = GymWrapper(env)
    env.reset()
    env.viewer.set_camera(camera_id=0)
    check_env(env, warn=True)
    env.reset(seed=0)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(t,": ",observation)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break