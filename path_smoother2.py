import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os

# 设置输入输出文件路径
input_csv = 'town5_waypoints_long_r0d2.csv'  # ✅ 你的输入文件
output_csv = 'town5_long_r0d2.csv'  # ✅ 保存带曲率的新文件

# 加载 CSV 文件
waypoints = pd.read_csv(input_csv)

# 提取坐标
x = waypoints['location_x'].values
y = waypoints['location_y'].values

# 使用一元样条函数进行平滑处理
smoothing_factor = 0.001
spl_x = UnivariateSpline(np.arange(len(x)), x, s=smoothing_factor)
spl_y = UnivariateSpline(np.arange(len(y)), y, s=smoothing_factor)

smoothed_x = spl_x(np.arange(len(x)))
smoothed_y = spl_y(np.arange(len(y)))

# 计算一阶导数与二阶导数
dx = np.gradient(smoothed_x)
dy = np.gradient(smoothed_y)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

# 计算曲率公式
curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5

# 生成 DataFrame 并保存为新 CSV
df = pd.DataFrame({
    'smoothed_x': smoothed_x,
    'smoothed_y': smoothed_y,
    'curvature': curvature
})
df.to_csv(output_csv, index=False)

print(f"✅ 曲率计算完成，结果已保存为: {output_csv}")

# 可视化结果（可选）
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='Original')
plt.plot(smoothed_x, smoothed_y, 'b-', label='Smoothed')
plt.title('Smoothed Trajectory with Curvature')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
