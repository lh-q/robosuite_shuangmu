import sys
sys.path.insert(0, "/home/lh/save2/robosuite")
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

    with open('robosuite-myframe/config/MyAssembly_FG.yml', 'r') as file:
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
    print(env.sim.model.body_names)
    env = DomainRandomizationWrapper(
        env,
        dynamics_randomization_args=my_dynamics_args,
        randomize_on_reset=True,
        randomize_every_n_steps = 300,

        )
    
    gym_env = GymWrapper(env=env)

    mon_env = Monitor(gym_env)
   
    # # ======== VecEnv 封装（重要） ========
    # vec_env = DummyVecEnv([lambda: mon_env])

    # # ======== 视频录制器 ========
    # vec_env = VecVideoRecorder(
    #     vec_env,
    #     video_folder="videos/",
    #     record_video_trigger=lambda step: step == 0,  # 第一次 reset 时开始录制
    #     video_length=1000,     # 录制多少 step（你可以改）
    #     name_prefix="eval_video",
    # )
                   
    # Save the trained model
    # model= SAC.load("robosuite-myframe/models/best_model_4lay_basenoise3.zip",env=mon_env)
    model= SAC.load("/home/lh/save2/my_assembly_SAC_0518.zip",env=mon_env)
    # model= SAC.load("robosuite-myframe/my_assembly",env=mon_env)
    # model= SAC.load("eval_logs/best_model",env=mon_env)
    # model= SAC.load("my_assembly_CG.zip",env=mon_env)
    # Optionally: evaluate the trained model
    #print(model.policy)
    #mean_reward, std_reward = evaluate_policy(model, mon_env, n_eval_episodes=5)
    #print(f"Loaded model reward: {mean_reward}, Std: {std_reward}")
    log_file = open("simulation_log.txt", "w")
    play_epsoide = 1
    while play_epsoide >0:
        play_epsoide -= 1
        obs,_= mon_env.reset()
        done = False
        i=0
        last_action = np.zeros(6)
        while not done:
            i+=1
            
            action, _states = model.predict(obs, deterministic=True)
            # act = 0.9*action + 0.1*last_action
            obs, rewards, terminated,truncated, info = mon_env.step(action)
            last_action = action

            log_file.write(f"Step {i}:\n")
            log_file.write(f"Rewards: {rewards}\n")
            log_file.write(f"Obs: {obs}\n")
            log_file.write(f"Action: {action}\n")
            log_file.write("\n")
            #print("step:", i)
            #print("rewards:", rewards)
            #print("obs:",obs)
            #print("action:",action)

            done = terminated or truncated 
            mon_env.render() 
            
    
    # Optionally: close the environment
    log_file.close()
    mon_env.close()
