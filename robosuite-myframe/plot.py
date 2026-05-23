
import matplotlib.pyplot as plt
import numpy as np
import parse_log_data as parse


def plot_data(obs_data, action_data):
    if not obs_data or not action_data:
        print("No Obs or Action data found in the file.")
        return

    obs_data = [np.array(x) for x in zip(*obs_data)]  # 转置 Obs 数据并转换为 numpy 数组
    action_data = [np.array(x) for x in zip(*action_data)]  # 转置 Action 数据并转换为 numpy 数组

    # 创建 3 行 2 列的子图
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    left_pairs = [
        [0, 9, 0],
        [1, 10, 1],
        [2, 11, 2],
    ]

    right_pairs = [
        [3, 6, 3],
        [4, 7, 4],
        [5, 8, 5],
    ]

    for idx, (obs_idx_1, obs_idx_2, action_idx) in enumerate(left_pairs):
        axs[idx, 0].plot(obs_data[obs_idx_1]*1000, label=f"Obs {obs_idx_1}")
        axs[idx, 0].plot(obs_data[obs_idx_2], label=f"Obs {obs_idx_2}")
        axs[idx, 0].plot(action_data[action_idx]*50, label=f"Action {action_idx}")
        axs[idx, 0].set_title(f"Obs {obs_idx_1}, Obs {obs_idx_2}, Action {action_idx}")
        axs[idx, 0].set_xlim(0, 300)  # 限制横坐标范围为 0-300
        
        axs[idx, 0].legend()
        axs[idx, 0].grid()
  

    for idx, (obs_idx_1, obs_idx_2, action_idx) in enumerate(right_pairs):
        axs[idx, 1].plot(obs_data[obs_idx_1]/np.pi*180, label=f"Obs {obs_idx_1}")
        axs[idx, 1].plot(obs_data[obs_idx_2], label=f"Obs {obs_idx_2}")
        axs[idx, 1].plot(action_data[action_idx]*50, label=f"Action {action_idx}")
        axs[idx, 1].set_title(f"Obs {obs_idx_1}, Obs {obs_idx_2}, Action {action_idx}")
        axs[idx, 1].set_xlim(0, 300)  # 限制横坐标范围为 0-300
        axs[idx, 1].legend()
        axs[idx, 1].grid()
      
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "simulation_log.txt"  # 替换为你的日志文件路径
    _,_,obs_data, action_data = parse.parse_data_from_txt(file_path)
    plot_data(obs_data, action_data)
    