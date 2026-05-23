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

from scipy.spatial.transform import Rotation as R

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
    obs = env.reset()
    # 目标位姿（假设无噪声）
    hole_quat= obs["hole_in_gripper_axisangle"]
    print(hole_quat)
    hole_pos = obs["hole_in_gripper_pos"]
    flag_sucess = False
    # print(env.sim.model.body_names)
    # env = DomainRandomizationWrapper(
    #     env,
    #     dynamics_randomization_args=my_dynamics_args,
    #     randomize_on_reset=True,
    #     randomize_every_n_steps = 300,

    #     )
    
    # gym_env = GymWrapper(env=env)
    # mon_env = Monitor(gym_env)
    # === 定义全局变量 ===
    fz_filtered = 0.0  # 初始化力滤波器
    # ======================
    # 3. 阶段划分
    # ======================

    def impedance_step_with_force_torque_control(
        pos_gain, ori_gain,
        steps=100,
        phase="approach",
        fz_target=-10, fz_tol=5,
        torque_limit=0.5,
        flag_sucess = False
    ):
        """
        阶段阻抗控制 + 力 + 力矩闭环调节
        phase: "approach" | "align" | "insert"
        """
        # 力/力矩控制增益（根据稳定性调整）
        Kp_force = 0.01     # 力控制增益
        # Kp_torque = 0.002    # 力矩控制增益

        for _ in range(steps):
            obs = env._get_observations()

            # 位姿误差
            pos_err = obs["hole_in_gripper_pos"].copy()
            rotvec_err = obs["hole_in_gripper_axisangle"].copy()

            # 当前末端力 / 力矩
            fz = obs["robot0_ee_force"][2]
            torque = np.linalg.norm(obs["robot0_ee_torque"])

            # 初始化控制指令
            delta_pos = np.zeros(3)
            delta_ori = np.zeros(3)

            # ================== 阶段1：Approach ==================
            if phase == "approach":
                desired_rel_pos = np.array([0, 0, 0.03])  # 预留3cm
                delta_pos = desired_rel_pos - pos_err
                delta_ori = np.zeros(3)

            # ================== 阶段2：Align ==================
            elif phase == "align":
                delta_pos = np.zeros(3)
                delta_ori = -rotvec_err  # 只调姿态

            # ================== 阶段3：Insert ==================
            elif phase == "insert":
                global fz_filtered
                alpha = 0.2
                fz_filtered = fz_filtered * (1 - alpha) + alpha * fz

                # --- (1) XY位置闭环 ---
                delta_pos[0] = -pos_err[0]
                delta_pos[1] = -pos_err[1]

                # --- (2) 接触判断 ---
                contact_force = abs(fz_filtered)
                contact_threshold = 1   # 开始接触判定值（可调）
                contact_ratio = np.clip((contact_force - contact_threshold) / 10.0, 0, 1)
                # contact_ratio: 0 表示未接触，1 表示稳定接触

                # --- (3) 混合控制 ---
                # 位移控制项
                pos_term_z = - pos_err[2]
                # 力控制项
                force_error = fz_filtered - fz_target
                force_term_z = -0.05 * force_error

                # 动态混合
                delta_z = (1 - contact_ratio) * pos_term_z + contact_ratio * force_term_z

                # 限制速度范围
                delta_z = np.clip(delta_z, -0.05, 0.0005)
                if delta_z > 0.005:
                    delta_z = 0

                delta_pos[2] = delta_z

                # --- (4) 姿态控制 ---
                delta_ori = -0.8 * rotvec_err

                print(f"[Insert] fz:{fz_filtered:.2f} | contact_ratio:{contact_ratio:.2f} | Δz:{delta_z:.4f}")


            else:
                delta_pos = -pos_err
                delta_ori = -rotvec_err

            # ================== 阻抗式速度指令 ==================
            v_linear = pos_gain.dot(delta_pos)
            v_angular = ori_gain.dot(delta_ori)

            action = np.concatenate([v_linear, v_angular])
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(-action)
            env.render()
            print(f"phase:{phase} | fz:{fz:.2f} | τ:{torque} | ΔZ:{delta_pos[2]:.6f}")
            if if_sucess():
                flag_sucess = True
                return  flag_sucess
            


    def if_sucess():
        # Get the observable dictionary
        observables = env._get_observations()
        #print(observables)
        axisangle_obs= observables["hole_in_gripper_axisangle"]
        pos_obs = observables["hole_in_gripper_pos"]
        dist_input = np.linalg.norm(pos_obs)
        angle_input = np.linalg.norm(axisangle_obs)

        force_input = np.linalg.norm(observables["robot0_ee_force"])
        torque_input = np.linalg.norm(observables["robot0_ee_torque"])
        print("torque_input:",torque_input)
        dist_input = dist_input * 1000
        angle_input = angle_input /np.pi *180.0
        fz = observables["robot0_ee_force"][2]
        print("fz:",fz)
        angle_within_tolerance = 1
        dist_within_tolerance = 3
        if (dist_input<dist_within_tolerance) and (angle_input<angle_within_tolerance):
            if force_input < 20  and torque_input < 3:
                if fz > -20 and fz < -3:
                    return True
                else:
                    return False
            else:
                return False

        else:
            return False

    # ======================
    # 阶段 1：定位阶段
    # ======================
    print("阶段 1：定位阶段（移动到距离目标 5cm 处）")
    impedance_step_with_force_torque_control(np.diag([30, 30, 30]), np.diag([0, 0, 0]),steps=100, phase="approach",flag_sucess=False)

    # ======================
    # 阶段 2：姿态对准阶段
    # ======================
    print("阶段 2：姿态对准阶段（对齐枪口与孔）")
    # 保持位置不变，调整姿态对齐孔
    impedance_step_with_force_torque_control(np.diag([0, 0, 0]), np.diag([20, 20, 20]),steps=50, phase="align",flag_sucess=False)

    # ======================
    # 阶段 3：插入阶段
    # ======================
    print("阶段 3：插入阶段（沿Z轴推进）")
    # 从当前位置沿自身Z轴插入
    flag_sucess = impedance_step_with_force_torque_control(np.diag([20, 20, 50]), np.diag([10, 10, 10]),steps=350, phase="insert",fz_target=-5, torque_limit=2,flag_sucess=False)

    env.close()
    if(flag_sucess):
        print("对接成功，停止运动！")
    else:
        print("对接失败！！！！")
