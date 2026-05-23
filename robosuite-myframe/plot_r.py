import matplotlib.pyplot as plt
import numpy as np
import parse_log_data as parse

# 假设日志文件名为 "simulation_log.txt"
filename = "simulation_log.txt"
steps, rewards,_,_ = parse.parse_data_from_txt(filename)

# 计算总的奖励
total_rewards = sum(rewards)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, label='Rewards per step')
plt.xlabel('Step')
plt.ylabel('Rewards')
plt.title('Rewards over Steps')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()

# 输出总奖励
print("Total rewards:", total_rewards)
