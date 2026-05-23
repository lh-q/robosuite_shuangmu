import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
import pandas as pd

import parse_log_data as parse

def calculate_statistics(data):
    statistics = {}
    columns = [ 'mx', 'my', 'mz','fx', 'fy', 'fz']
    data_array = np.array(data)
    
    for i, col in enumerate(columns):
        col_data = data_array[:, i]
        statistics[col] = {
            'mean': np.mean(col_data),
            'median': np.median(col_data),
            'std_dev': np.std(col_data),
            'max': np.max(col_data),
            'min': np.min(col_data),
            'Skewness': scistats.skew(col_data),
            '峰度/Kurtosis':scistats.kurtosis(col_data)
        }
    
    return statistics
def remove_outliers_iqr(data, iqr_scale=1.5):
    """
    使用 IQR 方法移除 sensor_data 中的尖端值
    
    参数：
        data: List[List[float]] - 原始传感器数据 (每行是 [mx, my, mz, fx, fy, fz])
        iqr_scale: float - IQR 放缩系数，通常为 1.5
    
    返回：
        filtered_data: List[List[float]] - 剔除尖端值后的数据
    """
    data_array = np.array(data)
    Q1 = np.percentile(data_array, 25, axis=0)
    Q3 = np.percentile(data_array, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_scale * IQR
    upper_bound = Q3 + iqr_scale * IQR

    mask = np.all((data_array >= lower_bound) & (data_array <= upper_bound), axis=1)
    return data_array[mask].tolist()

def plot_data_distributions(data):
    columns = [ 'mx', 'my', 'mz','fx', 'fy', 'fz']
    data_array = np.array(data)
    
    for i, col in enumerate(columns):
        col_data = data_array[:, i]
        time = np.arange(len(col_data))  # 假设时间序列为0到N-1
        
        plt.figure(figsize=(18, 5))
        
        # 绘制散点图
        plt.subplot(1, 2, 1)
        plt.scatter(time, col_data, color='blue', alpha=0.7)
        plt.title(f'{col} - Scatter Plot')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        # 绘制直方图
        plt.subplot(1, 2, 2)
        plt.hist(col_data, bins=30,density=True, color='green', alpha=0.7)
        plt.title(f'{col} - Histogram')
        # 拟合正态分布
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        mean = np.mean(col_data)
        std = np.std(col_data)
        p = scistats.norm.pdf(x, mean, std)

        plt.plot(x, p, 'k', linewidth=2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        '''
        # 绘制箱线图
        plt.subplot(2, 2, 3)
        plt.boxplot(col_data, vert=False)
        plt.title(f'{col} - Box Plot')
        plt.xlabel('Value')
        
        # 绘制 Q-Q 图
        plt.subplot(2, 2, 4)
        scistats.probplot(col_data, dist="uniform", plot=plt)
        plt.title(f'{col} - Q-Q Plot')
        '''
        plt.tight_layout()
        #plt.show()
        # 保存图形为图片文件
        output_path = f'{col}_plots.png'
        plt.savefig(output_path)
        plt.close()

def compute_covariance(obs_data):
    """
    计算力 (fx, fy, fz) 和力矩 (mx, my, mz) 与位置 (x, y, z) 及角度 (ax, ay, az) 的协方差值
    
    参数：
        obs_data: List[List[float]] - 观测数据，每一行包含 12 个数据 (x, y, z, ax, ay, az, fx, fy, fz, mx, my, mz)
    
    返回：
        cov_values: dict - 包含 fx, fy, fz, mx, my, mz 与 (x, y, z, ax, ay, az) 计算得到的 6 个协方差值
    """
    obs_array = np.array(obs_data)  # 转换为 NumPy 数组
    
    position_angle = obs_array[:, 0:6]  # 位置和角度信息 (x, y, z, ax, ay, az)
    torques = obs_array[:, 6:9]  # 力信息 (fx, fy, fz)
    forces = obs_array[:, 9:12]  # 力矩信息 (mx, my, mz)
    
    cov_values = {}
    
    # 计算 fx, fy, fz 与 (x, y, z, ax, ay, az) 的协方差值
    for i, force_name in enumerate(["fx", "fy", "fz"]):
        cov_matrix = np.cov(forces[:, i], position_angle, rowvar=False)
        cov_values[force_name] = cov_matrix[0, 1:]
    
    # 计算 mx, my, mz 与 (x, y, z, ax, ay, az) 的协方差值
    for i, torque_name in enumerate(["mx", "my", "mz"]):
        cov_matrix = np.cov(torques[:, i], position_angle, rowvar=False)
        cov_values[torque_name] = cov_matrix[0, 1:]
    
    return cov_values

def preprocess_obs_data(obs_data):
    """
    对 obs_data 进行预处理：
    - 位置 (x, y, z) 乘以 1000，单位转换为 mm
    - 角度 (ax, ay, az) 由弧度转换为角度 (度)
    
    参数：
        obs_data: List[List[float]] - 观测数据，每一行包含 12 个数据 (x, y, z, ax, ay, az, fx, fy, fz, mx, my, mz)
    
    返回：
        obs_data: List[List[float]] - 预处理后的数据
    """
    obs_array = np.array(obs_data)
    obs_array[:, 0:3] *= 1000  # 位置转换为 mm
    obs_array[:, 3:6] = obs_array[:, 3:6] / np.pi * 180  # 角度转换为度
    return obs_array.tolist()

if __name__ == "__main__":
    file_path = "simulation_log.txt"  # 替换为你的日志文件路径
    _,_,obs_data, action_data = parse.parse_data_from_txt(file_path)
    
    sensor_data = [obs[6:12] for obs in obs_data]  # 力信息
    #sensor_data = remove_outliers_iqr(sensor_data) 
    statistics = calculate_statistics(sensor_data)
    # 转换数据为 DataFrame
    df = pd.DataFrame(statistics)
    # 调整列顺序
    df = df[["fx", "fy", "fz", "mx", "my", "mz"]]
    # 打印表格
    print(df)
    
    obs_data = preprocess_obs_data(obs_data)  # 预处理数据
    cov_values = compute_covariance(obs_data)
    # 定义行索引
    row_labels = ["x", "y", "z", "ax", "ay", "az"]
    # 转换为 DataFrame
    df = pd.DataFrame(cov_values, index=row_labels)
    # 调整列顺
    df = df[["fx", "fy", "fz", "mx", "my", "mz"]]
    # 打印表格
    print(df)


    # 绘制数据分布图
    plot_data_distributions(sensor_data)

