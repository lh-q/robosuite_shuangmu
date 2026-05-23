import sys
import os
sys.path.insert(0, "/home/lh/save2/robosuite")
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import yaml
import imageio
from tqdm import tqdm

# 解决可能的递归问题
sys.setrecursionlimit(10000)

if __name__ == "__main__":
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # ===== 加载配置 =====
    # with open('robosuite-myframe/config/MyAssembly_FG.yml', 'r') as file:
    #     config = yaml.safe_load(file)

    # env_options = config["env"]
    # sac_options = config["SAC"]



    with open('robosuite-myframe/config/MyAssembly_TD3.yml', 'r') as file:
        config = yaml.safe_load(file)

    env_options = config["env"]
    sac_options = config["TD3"]

    # with open('robosuite-myframe/config/MyAssembly_PPO.yml', 'r') as file:
    #     config = yaml.safe_load(file)

    # env_options = config["env"]
    # sac_options = config["PPO"]


    # 加载控制器配置
    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller=controller)

    # ===== 创建视频保存目录 =====
    video_folder = "videos/"
    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, "eval_video.mp4")

    # ===== 创建Robosuite环境（最小化包装）=====
    env = suite.make(
        **env_options,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        render_camera="frontview",
        camera_names=["frontview"],
        camera_heights=1080,
        camera_widths=1920,
        controller_configs=controller_config,
    )

    # 包装为Gym环境
    gym_env = GymWrapper(env)
    gym_env.render_mode = "rgb_array"
    gym_env.metadata = {"render_fps": 30}

    # 可选：Monitor包装
    gym_env = Monitor(gym_env)

    # ===== 加载模型 =====
    # model = SAC.load(
    #     "robosuite-myframe/models/best_model_4lay_basenoise3.zip",
    #     env=gym_env,
    #     verbose=0
    # )

    model = TD3.load(
        "my_assembly_td3_best.zip",
        env=gym_env,
        verbose=0
    )

    # model = PPO.load(
    #     "eval_logs_ppo/best_model.zip",
    #     env=gym_env,
    #     verbose=0
    # )


    # ===== 运行并手动捕获帧 =====
    # 关键1：处理Gym环境的reset返回值，只取obs部分
    obs, info = gym_env.reset()  # Gym API：返回(obs, info)元组
    frames = []
    total_steps = 300

    for step in tqdm(range(total_steps), desc="Recording video"):
        # 关键2：为VecEnv训练的模型添加batch维度（形状从(n_features,)变为(1, n_features)）
        obs_batch = np.expand_dims(obs, axis=0)
        # 模型预测（使用batch后的obs）
        action, _states = model.predict(obs_batch, deterministic=True)
        # 关键3：去掉action的batch维度（从(1, n_actions)变为(n_actions,)）
        action = np.squeeze(action, axis=0)

        # 执行动作（处理Gym环境的step返回值）
        # 兼容Gymnasium新旧版本：新版返回(obs, reward, terminated, truncated, info)，旧版返回(obs, reward, done, info)
        step_result = gym_env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        # 从Robosuite原生环境获取渲染帧
        frame = env.sim.render(
            camera_name="frontview",
            width=1920,
            height=1080,
            depth=False
        )
        frames.append(np.flipud(frame))

        # 环境结束则重置
        if done:
            obs, info = gym_env.reset()  # 重置时同样只取obs

    # ===== 生成视频 =====
    if frames:
        imageio.mimsave(
            video_path,
            frames,
            fps=10,
            quality=10,
            codec="libx264",  # 指定编码器，确保MP4格式兼容
            ffmpeg_params=[           # 关键：CRF 18 视觉无损，慢但清晰
        '-crf', '18',
        '-preset', 'slower'
    ]
        )
        print(f"✅ 视频生成成功！保存路径：{video_path}")
        print(f"📹 视频总帧数：{len(frames)}，时长：{len(frames)/10:.2f}秒")
    else:
        print("❌ 未捕获到任何帧，视频生成失败")

    # ===== 清理环境 =====
    gym_env.close()
    env.close()
