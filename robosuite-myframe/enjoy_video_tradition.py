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
import imageio
import os
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from robosuite.wrappers import DomainRandomizationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    
    # 打印欢迎信息
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # 加载配置文件
    with open('robosuite-myframe/config/MyAssembly_FG.yml', 'r') as file:
        config = yaml.safe_load(file)
    env_options = config["env"]
    sac_options = config["SAC"]
    my_dynamics_args = config["MY_DYNAMICS_ARGS"]

    # 视频保存设置
    video_folder = "control_videos/"
    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, "impedance_control_video.mp4")
    frames = []  # 存储视频帧

    # 加载控制器配置
    controller = env_options.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    
    # 创建环境（同时开启屏幕渲染和离屏渲染）
    env = suite.make(
        **env_options,
        has_renderer=True,  # 保留屏幕渲染用于实时观察
        has_offscreen_renderer=True,  # 开启离屏渲染用于视频捕获
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        render_camera="frontview",  # 指定渲染相机
        camera_names=["frontview"],
        camera_heights=1080,
        camera_widths=1920,
        controller_configs=controller_config,
    )
    obs = env.reset()
    
    # 目标位姿初始化
    hole_quat = obs["hole_in_gripper_axisangle"]
    print(hole_quat)
    hole_pos = obs["hole_in_gripper_pos"]
    flag_sucess = False

    # 力滤波器初始化
    fz_filtered = 0.0

    def impedance_step_with_force_torque_control(
        pos_gain, ori_gain,
        steps=100,
        phase="approach",
        fz_target=-10, fz_tol=5,
        torque_limit=0.5,
        flag_sucess=False
    ):
        """阶段阻抗控制 + 力 + 力矩闭环调节，新增视频帧捕获"""
        Kp_force = 0.01  # 力控制增益
        
        # 使用tqdm显示进度并捕获帧
        for _ in tqdm(range(steps), desc=f"Phase: {phase}"):
            obs = env._get_observations()

            # 位姿误差计算
            pos_err = obs["hole_in_gripper_pos"].copy()
            rotvec_err = obs["hole_in_gripper_axisangle"].copy()

            # 当前末端力/力矩
            fz = obs["robot0_ee_force"][2]
            torque = np.linalg.norm(obs["robot0_ee_torque"])

            # 初始化控制指令
            delta_pos = np.zeros(3)
            delta_ori = np.zeros(3)

            # 阶段1：Approach
            if phase == "approach":
                desired_rel_pos = np.array([0, 0, 0.03])  # 预留3cm
                delta_pos = desired_rel_pos - pos_err
                delta_ori = np.zeros(3)

            # 阶段2：Align
            elif phase == "align":
                delta_pos = np.zeros(3)
                delta_ori = -rotvec_err  # 只调姿态

            # 阶段3：Insert
            elif phase == "insert":
                global fz_filtered
                alpha = 0.2
                fz_filtered = fz_filtered * (1 - alpha) + alpha * fz

                # XY位置闭环
                delta_pos[0] = -pos_err[0]
                delta_pos[1] = -pos_err[1]

                # 接触判断
                contact_force = abs(fz_filtered)
                contact_threshold = 1
                contact_ratio = np.clip((contact_force - contact_threshold) / 10.0, 0, 1)

                # 混合控制
                pos_term_z = - pos_err[2]
                force_error = fz_filtered - fz_target
                force_term_z = -0.05 * force_error
                delta_z = (1 - contact_ratio) * pos_term_z + contact_ratio * force_term_z
                delta_z = np.clip(delta_z, -0.05, 0.0005)
                if delta_z > 0.005:
                    delta_z = 0
                delta_pos[2] = delta_z

                # 姿态控制
                delta_ori = -0.8 * rotvec_err

                print(f"[Insert] fz:{fz_filtered:.2f} | contact_ratio:{contact_ratio:.2f} | Δz:{delta_z:.4f}")

            else:
                delta_pos = -pos_err
                delta_ori = -rotvec_err

            # 阻抗式速度指令
            v_linear = pos_gain.dot(delta_pos)
            v_angular = ori_gain.dot(delta_ori)
            action = np.concatenate([v_linear, v_angular])
            action = np.clip(action, -1, 1)

            # 执行动作
            obs, reward, done, info = env.step(-action)
            env.render()  # 屏幕渲染

            # 捕获当前帧用于视频
            frame = env.sim.render(
                camera_name="frontview",
                width=1920,
                height=1080,
                depth=False
            )
            frames.append(np.flipud(frame))  # 翻转帧以正确显示

            # 检查是否成功
            if if_sucess():
                flag_sucess = True
                return flag_sucess

        return flag_sucess


    def if_sucess():
        """判断是否成功对接"""
        observables = env._get_observations()
        axisangle_obs = observables["hole_in_gripper_axisangle"]
        pos_obs = observables["hole_in_gripper_pos"]
        dist_input = np.linalg.norm(pos_obs)
        angle_input = np.linalg.norm(axisangle_obs)

        force_input = np.linalg.norm(observables["robot0_ee_force"])
        torque_input = np.linalg.norm(observables["robot0_ee_torque"])
        dist_input = dist_input * 1000
        angle_input = angle_input / np.pi * 180.0
        fz = observables["robot0_ee_force"][2]

        angle_within_tolerance = 1
        dist_within_tolerance = 3
        if (dist_input < dist_within_tolerance) and (angle_input < angle_within_tolerance):
            if force_input < 20 and torque_input < 3:
                if fz > -20 and fz < -3:
                    return True
        return False


    # 执行控制流程
    print("阶段 1：定位阶段（移动到距离目标 5cm 处）")
    flag_sucess = impedance_step_with_force_torque_control(
        np.diag([30, 30, 30]), 
        np.diag([0, 0, 0]),
        steps=100, 
        phase="approach"
    )

    if not flag_sucess:
        print("阶段 2：姿态对准阶段（对齐枪口与孔）")
        flag_sucess = impedance_step_with_force_torque_control(
            np.diag([0, 0, 0]), 
            np.diag([20, 20, 20]),
            steps=50, 
            phase="align"
        )

    if not flag_sucess:
        print("阶段 3：插入阶段（沿Z轴推进）")
        flag_sucess = impedance_step_with_force_torque_control(
            np.diag([20, 20, 50]), 
            np.diag([10, 10, 10]),
            steps=350, 
            phase="insert",
            fz_target=-5, 
            torque_limit=2
        )

    # 生成视频
    if frames:
        imageio.mimsave(
            video_path,
            frames,
            fps=10,
            quality=10,
            codec="libx264",
            ffmpeg_params=['-crf', '18', '-preset', 'slower']
        )
        print(f"✅ 视频生成成功！保存路径：{video_path}")
        print(f"📹 视频总帧数：{len(frames)}，时长：{len(frames)/10:.2f}秒")
    else:
        print("❌ 未捕获到任何帧，视频生成失败")

    # 关闭环境
    env.close()
    if flag_sucess:
        print("对接成功，停止运动！")
    else:
        print("对接失败！！！！")
