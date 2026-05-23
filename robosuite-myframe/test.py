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

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import time

import numpy as np
if __name__ == "__main__":
    '''
        这个py是用来第一次训练模型的
    '''    
    # print welcome info
    #print("Welcome to robosuite v{}!".format(suite.__version__))
    #print(suite.__logo__)

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
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        
        controller_configs=controller_config,
    )
        # 2. 获取完整 XML
    xml_str = env.sim.model.get_xml()   #  这是已经合并完的完整 XML 字符串
    # 3. 保存到文件（方便检查最终加载的模型）
    with open("merged_env.xml", "w") as f:
        f.write(xml_str)
    obs = env.reset()
    obs,r,_,_ = env.step(np.zeros(6))
    hole_frame_2_id = env.sim.model.body_name2id("fuelinghole3")
    time.sleep(0.02)
    # print("angle:",obs["gripper_to_hole_axisangle_real"])
    robot = env.robots[0]  # 获取第一个机器人
    joint_names = robot.joint_indexes # 机器人所有关节名称
    print("机器人关节名称列表：", joint_names)
    for step in range(500):
    #    print("eef",obs["robot0_eef_pos"])  
    #    print("eef",obs["robot0_eef_quat"]) 
        
       #print(obs["hole_in_gripper_axisangle"]) 
       #print(obs["hole_quat"])
    #    obs,r,_,_ = env.step([0.01,0.01,0.01,0.01,0.01,0.01])
     # 读取 fuelinghole3/2 的位置
     # 读取关节角
       joint_angles = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
       hole_frame_2_pos = env.sim.data.xpos[hole_frame_2_id].copy()
       parent_quat = env.sim.data.xquat[hole_frame_2_id].copy()
       obs,r,_,_ = env.step([0,0.0,1.00,0.0,0.0,0.0])
    #    print(f"\nStep {step} - 位置 (x,y,z): {hole_frame_2_pos.round(6)} 米")
    #    print(f"\nStep {step} - fuelinghole3 姿态（四元数）：{parent_quat.round(6)}")
    #    print(f"\n:{joint_angles.round(6)*180/np.pi}")
    #    print("hole_in_gripper_pos:",obs["hole_in_gripper_pos"])
    #    print(obs["hole_in_gripper_pos"])
    #    print("angle:",obs["hole_in_gripper_axisangle"])
    #    obs,r,_,_ = env.step(np.zeros(6))
    #    print("force_x",obs["robot0_ee_force"][0])
    #    print("force_y",obs["robot0_ee_force"][1])
    #    print("force_z",obs["robot0_ee_force"][2])
       print("force:",obs["robot0_ee_force"])
    #    print("torque:",obs["robot0_ee_torque"])
    #    print(obs["gripper_to_hole_pos_real"])
    #    print(obs["gripper_to_hole_pos_noise"]) 
    #    print("angle:",obs["gripper_to_hole_axisangle_real"])
    #    print("eef_pos:", obs["robot0_eef_pos"])
    #    print("hole_pos:", obs["hole_pos"])

       env.render()
       time.sleep(0.02)