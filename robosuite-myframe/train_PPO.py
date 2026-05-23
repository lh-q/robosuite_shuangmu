import sys
sys.path.insert(0, "/home/lh/save2/robosuite")

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
from robosuite.wrappers import DomainRandomizationWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

import yaml


if __name__ == "__main__":

    # 读取配置
    with open('robosuite-myframe/config/MyAssembly_PPO.yml', 'r') as file:
        config = yaml.safe_load(file)

    env_options = config["env"]
    ppo_options = config["PPO"]
    my_dynamics_args = config["MY_DYNAMICS_ARGS"]

    # 加载 controller
    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller=controller)

    # 创建 robosuite 环境
    env = suite.make(
        **env_options,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
    )

    # 可选：Domain Randomization
    # env = DomainRandomizationWrapper(
    #     env,
    #     randomize_color=False,
    #     randomize_camera=False,
    #     randomize_lighting=False,
    #     randomize_dynamics=True,
    #     dynamics_randomization_args=my_dynamics_args,
    #     randomize_on_reset=True,
    #     randomize_every_n_steps=300,
    # )

    # === Gym Wrapper ===
    gym_env = GymWrapper(env)
    mon_env = Monitor(gym_env)

    # === Eval Callback ===
    eval_log_dir = "./eval_logs_ppo/"
    eval_callback = EvalCallback(
        mon_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # ====== 创建 PPO 模型 ======
    # 关键修复：使用 env=mon_env
    model = PPO(env=mon_env, **ppo_options)
    # model = PPO.load("my_assembly_ppo.zip", env=mon_env)

    # ====== 训练前评估 ======
    mean_reward, std_reward = evaluate_policy(model, mon_env, n_eval_episodes=5)
    print(f"Initial model reward: {mean_reward}, Std: {std_reward}")

    # ====== 开始训练 ======
    model.learn(
        total_timesteps=2000000,
        progress_bar=True,
        callback=eval_callback
    )

    # ====== 保存 ======
    model.save("my_assembly_ppo_1212")

    gym_env.close()
