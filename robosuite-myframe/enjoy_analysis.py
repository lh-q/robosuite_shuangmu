import sys
sys.path.insert(0, "/home/lh/robosuite")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import yaml


from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from robosuite.wrappers import DomainRandomizationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    
    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    with open('config/MyAssembly.yml', 'r') as file:
        config = yaml.safe_load(file)
    # Environment configuration
    env_options = config["env"]
    sac_options = config["SAC"]
    my_dynamics_args = config["MY_DYNAMICS_ARGS"]

    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller= controller)
    env = suite.make(
        **env_options,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        #hard_reset=False,  # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
        controller_configs=controller_config,
    )
    
    env = DomainRandomizationWrapper(
        env,
        dynamics_randomization_args=my_dynamics_args,
        randomize_on_reset=True,
        randomize_every_n_steps = 300,

        )
    
    gym_env = GymWrapper(env=env)

    mon_env = Monitor(gym_env)
   
    
                   
    # Save the trained model
    model= SAC.load("models/best_model_4lay_basenoise5.zip",env=mon_env)
    #model= SAC.load("my_assembly_nonoise",env=mon_env)
    #model= SAC.load("my_assembly",env=mon_env)
    #model= SAC.load("best_model",env=mon_env)
    log_file = open("simulation_log.txt", "w")
    play_epsoide = 100
    # 用于统计
    total_episodes = play_epsoide
    success_count = 0
    total_steps = 0
    max_force_all = []
    max_torque_all = []
    episode_data = []

    while play_epsoide >0:
        play_epsoide -= 1
        obs,_= mon_env.reset()
                
        done = False
        i = 0
        max_force = 0
        max_torque = 0
        episode_success = False
        last_action = np.zeros(6)
        while not done:
            i+=1
            
            action, _states = model.predict(obs, deterministic=True)
            #act = 0.9*action + 0.1*last_action
            obs, rewards, terminated,truncated, info = mon_env.step(action)
            last_action = action
            # 判断是否成功（奖励为1则表示成功）
            if abs(rewards - 1) < 0.01:
                terminated = True
            # 更新最大力和力矩（取绝对值再比较）
            max_force = max(max_force, max(abs(f) for f in obs[9:12]))
            max_torque = max(max_torque, max(abs(t) for t in obs[6:9]))
            if max_force > 200 or max_torque > 20:
                truncated = True
            log_file.write(f"Step {i}:\n")
            log_file.write(f"Rewards: {rewards}\n")
            log_file.write(f"Obs: {obs}\n")
            log_file.write(f"Action: {action}\n")
            log_file.write("\n")

            done = terminated or truncated 
            #mon_env.render() 

        # 更新统计
        if terminated:
            success_count += 1
            
        total_steps += i
        
        max_force_all.append(max_force)
        max_torque_all.append(max_torque)
    
    success_rate = success_count / total_episodes
    average_steps = total_steps / total_episodes
    print("\n--- Simulation Summary ---")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Steps: {average_steps:.2f}")
    print(f"Max Force (all episodes): {max(max_force_all):.2f}")
    print(f"Max Torque (all episodes): {max(max_torque_all):.2f}")
    #print("force",max_force_all)
    #print("torque",max_torque_all)
    mon_env.close()
    log_file.close()