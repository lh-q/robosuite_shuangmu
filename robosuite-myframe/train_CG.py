import sys
sys.path.insert(0, "/home/lh/save/robosuite")
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

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from obs_logger_callback import ObsLoggerCallback
from stable_baselines3.common.callbacks import CallbackList
if __name__ == "__main__":
    '''
        这个py是用来第一次训练模型的
    '''    
    # print welcome info
    #print("Welcome to robosuite v{}!".format(suite.__version__))
    #print(suite.__logo__)

    with open('robosuite-myframe/config/MyAssembly_CG.yml', 'r') as file:
        config = yaml.safe_load(file)
    # Environment configuration
    env_options = config["env"]
    sac_options = config["SAC"]
    my_dynamics_args = config["MY_DYNAMICS_ARGS"]

    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller= controller)
    env = suite.make(
        **env_options,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        
        controller_configs=controller_config,
    )
    obs = env.reset()
    # obs,r,_,_ = env.step(np.zeros(6))

    # print("axisangle_real:", obs["gripper_to_hole_axisangle_real"])
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False, # randomize_color currently only works for mujoco==3.1.1
        randomize_camera=False,
        randomize_lighting=False,
        randomize_dynamics=True,
        dynamics_randomization_args=my_dynamics_args,
        randomize_on_reset=True,
        randomize_every_n_steps = 300,
        )

    gym_env = GymWrapper(env=env)
    '''
    #这个部分是为了拿到观测器里面都有哪些观测量并且他们的维度是啥
    obs = env._get_observations()
    keys, sizes = [], []
    for k, v in obs.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            # 确保是可迭代类型
            sizes.append(np.array(v).size)
        else:
            sizes.append(1)
        keys.append(k)

    print("Flatten 顺序:")
    for i, (k, s) in enumerate(zip(keys, sizes)):
        print(f"{i:02d}: {k} ({s})")
    '''
    mon_env = Monitor(gym_env)
    obs = mon_env.reset()
    # Check if the environment is valid
    #check_env(gym_env, warn=True)    #resulet  is true 
    eval_log_dir = "./eval_logs/"
    #每隔5000步评估一次， 每次评估五个回合
    eval_callback = EvalCallback(mon_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=10000,
                              n_eval_episodes=5, deterministic=True,
                              render=False)
    sac_options["env"] = mon_env  # 将环境实例传递给 sac_options
    # Set up SAC model
    # 定义自定义观察记录 callback，每10步记录一次
    # callback = CallbackList([eval_callback, ObsLoggerCallback()])

    # model = SAC(**sac_options ) #从头训练 用这个
    #model= SAC.load("models/best_model_4lay_basenoise2.zip",env=mon_env)
    #model= SAC.load("best_model",env=mon_env)
    #model = SAC.load("my_assembly_nonoise",env= mon_env) #继续训练用这个
    # model = SAC.load("my_assembly",env= mon_env) #继续训练用这个
    #model.load_replay_buffer("my_assembly_buffer.pkl")
    model= SAC.load("my_assembly_CG_test.zip",env=mon_env)
    mean_reward, std_reward = evaluate_policy(model, mon_env, n_eval_episodes=10)
    print(f"Loaded model reward: {mean_reward}, Std: {std_reward}")
    # Train the model
    model.learn(total_timesteps=1000000,progress_bar=True, callback=eval_callback,tb_log_name="sac_robosuite_run_CG_test")  # You can adjust this number based on your needs

    # Save the trained model
    model.save("my_assembly_CG_test")
    model.save_replay_buffer("my_assembly_buffer_CG_test")


    # Optionally: close the environment
    gym_env.close()
