import sys
sys.path.insert(0, "/home/lh/save2/robosuite")

import numpy as np
import yaml
import robosuite as suite
from robosuite.wrappers import GymWrapper, DomainRandomizationWrapper
from robosuite.controllers import load_controller_config
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":

    # === Load config ===
    with open("robosuite-myframe/config/MyAssembly_TD3.yml", "r") as f:
        config = yaml.safe_load(f)

    env_options = config["env"]
    td3_options = config["TD3"]
    my_dynamics_args = config["MY_DYNAMICS_ARGS"]

    # === Load controller ===
    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller=controller)

    # === Create environment ===
    env = suite.make(
        **env_options,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
    )

    # === Apply domain randomization ===
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

    # === Setup TD3 action noise ===
    n_actions = mon_env.action_space.shape[0]
    mean = np.ones(n_actions) * td3_options.pop("action_noise_mean")
    sigma = np.ones(n_actions) * td3_options.pop("action_noise_sigma")
    td3_options["action_noise"] = NormalActionNoise(mean=mean, sigma=sigma)

    # === Eval callback ===
    eval_log_dir = "./eval_logs_td3_best/"
    eval_callback = EvalCallback(
        mon_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # === Create TD3 model ===
    model = TD3(env=mon_env, **td3_options)
    # model = TD3(env=mon_env, **td3_options)
    # === Evaluate before training ===
    mean_reward, std_reward = evaluate_policy(model, mon_env, n_eval_episodes=5)
    print(f"Initial model reward: {mean_reward:.4f}, Std: {std_reward:.4f}")

    # === Train TD3 ===
    model.learn(total_timesteps=2000000, progress_bar=True, callback=eval_callback)

    # === Save model & replay buffer ===
    model.save("my_assembly_ppo_best")
    model.save_replay_buffer("my_assembly_ppo_buffer_best")

    gym_env.close()
