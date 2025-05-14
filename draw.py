
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import ast
import pandas as pd
from scipy.interpolate import interp1d

def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std_dev]

def draw_latjerk(file_path):
    data = pd.read_csv(file_path)
# 定义去除离群值的函数
    def remove_outliers(data, threshold=3):
        mean = np.mean(data)
        std_dev = np.std(data)
        return [x for x in data if abs(x - mean) <= threshold * std_dev]

    # 过滤并转换为浮点数
    lateral_jerk_values = []
    for col in data.columns:
        try:
            value = float(col)
            lateral_jerk_values.append(value)
        except ValueError:
            continue
    # 去除离群值
    filtered_lateral_jerk_values = remove_outliers(lateral_jerk_values)

    # 生成新的x轴timesteps

    timesteps = list(range(len(filtered_lateral_jerk_values)))
    # 生成x轴timesteps

    # 确保数据不为空后绘制折线图
    if lateral_jerk_values:
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, filtered_lateral_jerk_values, label="Lateral Jerk (m^3)")
        plt.xlabel("Timesteps")
        plt.ylabel("Lateral Jerk (m^3)")
        plt.title("Lateral Jerk Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("数据为空，无法绘制图表，请检查文件内容。")

def load_and_process_other_files(directory):
    for filename in os.listdir(directory):
        if 'accel' in filename or 'speed' in filename or 'latjerk' in filename or 'lkd' in filename or 'deviation' in filename  and filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            data_list = load_list_from_csv(filepath)

            # 转换为 DataFrame
            df = pd.DataFrame(data_list, columns=['value'])

            # 保存为同名文件
            output_filepath = os.path.join(directory, filename)
            df.to_csv(output_filepath, index=False)

    # 保存为 CSV 文件
    csv_file_path = 'output.csv'  # 输出文件路径
    df.to_csv(csv_file_path, index=False)


def load_and_process_vpath_files(directory):
    """
    遍历指定文件夹，将文件名中包含 'vpath' 的文件内容恢复为二维列表，并保存为同名 CSV 文件。
    
    Args:
        directory (str): 要遍历的文件夹路径。
    """
    for filename in os.listdir(directory):
        if 'path' in filename and 'pp2.0' in filename and filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # 读取并处理文件
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                data = next(reader)
                # data_list = [item for item in data]
                # print(data_list)
                data_list = [ast.literal_eval(item) for item in data]

            # 转换为 DataFrame
            df = pd.DataFrame(data_list, columns=['x', 'y'])

            # 保存为同名文件
            output_filepath = os.path.join(directory, filename)
            df.to_csv(output_filepath, index=False)

def plot_comparison2(a1, a2, a3, xlabel=None, ylabel='', unit='', dpi=300, save=True):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(a1, label='Speed', color='green')
    ax1.set_xlabel(xlabel if xlabel else 'Time step', fontsize=14)
    ax1.set_ylabel(f'{ylabel} Speed (m/s)', fontsize=14, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    ax2 = ax1.twinx()
    ax2.plot(a2, label='Look-ahead', color='orange')
    ax2.set_ylabel(f'{ylabel} Look-ahead (m)', fontsize=14, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax3 = ax1.twinx()
    ax3.plot(a3, label='Deviation', color='gray')
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel(f'{ylabel} Deviation (m)', fontsize=14, color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')
    plt.grid(True)
    fig.tight_layout()
    
    if save:
        plt.savefig(f'{ylabel}.png', dpi=dpi)
    
    plt.show()

def load_list_from_csv(filepath):
    """
    从CSV文件恢复单行数据
    filepath (str): 文件路径，包括文件名
    返回:
    list: 恢复的列表
    """
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data_from_csv = next(reader)  # 读取一行
        return [float(i) for i in data_from_csv]  # 将字符串转换为整数
    

    
def read_file_to_list(file_path):
    '''
    将单列csv文件提取为列表
    '''
    returns_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        print(reader)
        for row in reader:
            returns_list.append(float(row[1]))
    return returns_list

def load_vpath_from_csv(filepath):
    '''
    将单行存储的字符串恢复为二维列表
    '''
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = next(reader)
        data_list = [ast.literal_eval(item) for item in data]
    return data_list

def moving_average(a, window_size=19):

    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))






def draw_path(file_path):
    data = pd.read_csv(file_path)
    x_coordinate = data['x'].to_list()
    y_coordinate = data['y'].to_list()
    path = [[x, y] for x, y in zip(x_coordinate, y_coordinate)]
    

    # 绘制图形
    plt.plot(x_coordinate, y_coordinate, color='red', label='vpath', lw=1)
    plt.legend()
    plt.grid(True)

    plt.show()
    

def draw_paths(file_paths):
    """
    绘制多个文件的路径到同一张图上。
    
    参数:
        file_paths (list): 文件路径列表，每个文件包含 'x' 和 'y' 列。
    """
    plt.figure(figsize=(10, 6))  # 设置画布大小
    for idx, file_path in enumerate(file_paths):
        # 读取数据
        agent_name = ''
        if 'ppo_compare_final' in file_path:
            agent_name = 'N-PPO'
        elif 'ppo_final' in file_path:
            agent_name = 'C-PPO'
        elif 'sac_compare_final' in file_path:
            agent_name = 'N-SAC'
        elif 'sac_final' in file_path:
            agent_name = 'C-SAC'
        elif 'pp1.5.csv' in file_path:
            agent_name = 'PP 1.5'
        elif 'Processed_Coordinates' in file_path:
            agent_name = 'Reference'
        data = pd.read_csv(file_path)
        x_coordinate = data['x'].to_list()
        y_coordinate = data['y'].to_list()
        
        # 绘制路径
        plt.plot(x_coordinate, y_coordinate, label=f'{agent_name} Path', lw=1)
    
    # 添加图例、网格和标题
    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=20)
    plt.ylabel('Y Coordinate', fontsize=20)
    plt.title('Multiple Paths', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.show()


def draw_paths2(file_paths):
    """
    绘制多个文件的路径到同一张图上，并让图例围绕图片。
    
    参数:
        file_paths (list): 文件路径列表，每个文件包含 'x' 和 'y' 列。
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # 设置画布大小

    # 颜色和线型映射
    line_styles = {
        'N-PPO': ('blue', '-'),
        'C-PPO': ('red', '--'),
        'N-SAC': ('green', '-.'),
        'C-SAC': ('purple', ':'),
        'PP 2.0': ('orange', '-'),
        'Reference': ('black', 'solid')
    }

    for file_path in file_paths:
        # 确定 agent 名称
        agent_name = ''
        if 'ppo_compare_final' in file_path:
            agent_name = 'N-PPO'
        elif 'ppo_final' in file_path:
            agent_name = 'C-PPO'
        elif 'sac_compare_final' in file_path:
            agent_name = 'N-SAC'
        elif 'sac_final' in file_path:
            agent_name = 'C-SAC'
        elif 'pp2.0.csv' in file_path:
            agent_name = 'PP 2.0'
        elif 'Processed_Coordinates' in file_path:
            agent_name = 'Reference'

        # 读取数据
        data = pd.read_csv(file_path)
        x_coordinate = data['x'].to_list()
        y_coordinate = data['y'].to_list()

        # 获取对应的颜色和线型
        color, linestyle = line_styles.get(agent_name, ('gray', '-'))

        # 绘制路径
        ax.plot(x_coordinate, y_coordinate, label=f'{agent_name}', color=color, linestyle=linestyle, lw=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 添加图例围绕图片
    legend = ax.legend(fontsize=24, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    # 调整图像边距，确保图例不会遮挡内容
    plt.grid(alpha=0.3)
    # plt.xlabel('X Coordinate', fontsize=14)
    # plt.ylabel('Y Coordinate', fontsize=14)
    plt.title('Paths with Different Agent', fontsize=24)
    # plt.tick_params(axis='both', which='major', labelsize=24)
    plt.subplots_adjust(bottom=0.2)  # 调整底部空间给图例

    plt.show()

def draw_compare_image(file_paths, xlabel='', ylabel='', unit='', title=''):
    """
    绘制多个文件到同一张图上。
    
    参数:
        file_paths (list): 文件路径列表，每个文件包含一个列表。
    """
    plt.figure(figsize=(10, 6))  # 设置画布大小
    for idx, file_path in enumerate(file_paths):
        # 读取数据
        agent_name = ''
        if 'ppo_compare_final' in file_path:
            agent_name = 'N-PPO'
        elif 'ppo_final' in file_path:
            agent_name = 'C-PPO'
        elif 'sac_compare_final' in file_path:
            agent_name = 'N-SAC'
        elif 'sac_final' in file_path:
            agent_name = 'C-SAC'
        elif 'pp1.5.csv' in file_path:
            agent_name = 'PP 1.5'
        elif 'pid' in file_path:
            agent_name = 'PID'
        data = pd.read_csv(file_path)
        x_coordinate = data['Episode'].to_list()
        y_coordinate = data['Return'].to_list()
        y_coordinate = remove_outliers(y_coordinate)
        # y_coordinate = moving_average(y_coordinate)

        print(f"({ylabel} {agent_name})max_value:{np.max(y_coordinate)}")
        
        # 绘制路径
        plt.plot(y_coordinate, label=f'{agent_name} {ylabel}', lw=1)
    
    # 添加图例、网格和标题
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel+unit)
    plt.title(title)
    plt.show()

def draw_aligned_compare_image(file_paths, xlabel='', ylabel='', unit='', title=''):
    """
    绘制多个文件到同一张图上。
    
    参数:
        file_paths (list): 文件路径列表，每个文件包含一个列表。
    """
    color_styles = {
        'N-PPO': ('blue'),
        'C-PPO': ('red'),
        'N-SAC': ('green'),
        'C-SAC': ('purple'),
        'PP 2.0': ('orange'),
        'Reference': ('black')
    }
    def find_shorter_longer_idx(plot_array):
        
        return np.argmax(plot_array), np.argmin(plot_array)
    plt.figure(figsize=(10, 6))  # 设置画布大小
    plot_array=[]
    plot_len=[]
    plot_dict={}
    agent_names = []
    for idx, file_path in enumerate(file_paths):
        # 读取数据
        agent_name = ''
        if 'ppo_compare_final' in file_path:
            agent_name = 'N-PPO'
        elif 'ppo_final' in file_path:
            agent_name = 'C-PPO'
        elif 'sac_compare_final' in file_path:
            agent_name = 'N-SAC'
        elif 'sac_final' in file_path:
            agent_name = 'C-SAC'
        elif 'pp1.0.csv' in file_path:
            agent_name = 'PP 1.0'
        elif 'pp1.5.csv' in file_path:
            agent_name = 'PP 1.5'
        elif 'pp2.0.csv' in file_path:
            agent_name = 'PP 2.0'
        elif 'pid' in file_path:
            agent_name = 'PID'
        data = pd.read_csv(file_path)

        y_coordinate = data['value'].to_list()
        # y_coordinate = remove_outliers(y_coordinate)
        # y_coordinate = moving_average(y_coordinate)
        plot_dict[agent_name] = y_coordinate
        agent_names.append(agent_name)
        print(f"({ylabel} {agent_name})max_value:{np.max(np.abs(y_coordinate))}")
        print(f"({ylabel} {agent_name})mean_value:{np.mean(np.abs(y_coordinate))}")
        
        # 生成对应的横坐标
    for name in agent_names:
        l=len(plot_dict[name])
        x = np.linspace(0, 1, l)  # 处理坐标范围
        plot_array.append(x)  # 添加到绘图使用的坐标范围
        plot_len.append(l) # 记录长度方便找到agent_name
    long_agent_idx,short_agent_idx = find_shorter_longer_idx(plot_len)
    # 对短数组进行插值放缩，使其横坐标与长数组对齐
    interp_func = interp1d(plot_array[short_agent_idx], plot_dict[agent_names[short_agent_idx]], kind='linear', fill_value="extrapolate")  # 插值函数
    y2_rescaled = interp_func(plot_array[long_agent_idx])  # 将短数组映射到长数组的横坐标范围
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(plot_array[long_agent_idx], plot_dict[agent_names[long_agent_idx]], label=agent_names[long_agent_idx], lw=1, color=color_styles.get(agent_names[long_agent_idx], ('gray')))  # 长数组
    plt.plot(plot_array[long_agent_idx], y2_rescaled, label=agent_names[short_agent_idx], lw=1, color=color_styles.get(agent_names[short_agent_idx], ('gray')))  # 放缩后的短数组
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel+unit, fontsize=24)
    plt.title(title, fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.show()



CSVFOLDER = 'csvfile/'

load_and_process_other_files(CSVFOLDER)
load_and_process_vpath_files(CSVFOLDER)
    

file_paths = ['csvfile/Processed_Coordinates.csv','csvfile/vpath0_pp2.0.csv','csvfile/final_vpath0_ppo_compare_final.csv','csvfile/final_vpath0_ppo_final.csv','csvfile/final_vpath0_sac_compare_final.csv','csvfile/final_vpath0_sac_final.csv']
training_episode_reward_paths = ['csvfile/ppo_compare_final.csv','csvfile/ppo_final.csv','csvfile/sac_compare_final.csv','csvfile/sac_final.csv']
lat_jeark_pp_sac = ['csvfile/latjerk3_pp2.0.csv','csvfile/final_latjerk5_sac_final.csv']
lat_jeark_ppo_sac = ['csvfile/final_latjerk0_ppo_final.csv','csvfile/final_latjerk0_sac_final.csv']
# draw_compare_image(training_episode_reward_paths,"Episodes", "Reward", title="Training Reward Comparison")
# draw_paths2(file_paths)
# draw_aligned_compare_image(lat_jeark_pp_sac, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of C-PPO and C-SAC")

# 奖励函数有效性
deviation_paths_sac = ['csvfile/final_deviation8_sac_final.csv','csvfile/final_deviation3_sac_compare_final.csv']
deviation_paths_ppo = ['csvfile/final_deviation1_ppo_final.csv','csvfile/final_deviation1_ppo_compare_final.csv']
# draw_aligned_compare_image(deviation_paths_sac, "Tracking Completion Rate", "Deviation", "(m)", "Deviation Comparison of N-SAC and C-SAC")
# draw_aligned_compare_image(deviation_paths_ppo, "Tracking Completion Rate", "Deviation", "(m)", "Deviation Comparison of N-PPO and C-PPO")

accel_paths_sac = ['csvfile/final_accel0_sac_final.csv','csvfile/final_accel1_sac_compare_final.csv']
accel_paths_ppo = ['csvfile/final_accel1_ppo_final.csv','csvfile/final_accel0_ppo_compare_final.csv']
# draw_aligned_compare_image(accel_paths_sac, "Tracking Completion Rate", "Lat-Acceleration", "(m/s^2)", "Lateral Acceleration Comparison of N-SAC and C-SAC")
# draw_aligned_compare_image(accel_paths_ppo, "Tracking Completion Rate", "Lat-Acceleration", "(m/s^2)", "Lateral Acceleration Comparison of N-PPO and C-PPO")

jerk_paths_sac = ['csvfile/final_latjerk5_sac_final.csv','csvfile/final_latjerk7_sac_compare_final.csv']
jerk_paths_ppo = ['csvfile/final_latjerk3_ppo_final.csv','csvfile/final_latjerk2_ppo_compare_final.csv']
# draw_aligned_compare_image(jerk_paths_sac, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of N-SAC and C-SAC")
# draw_aligned_compare_image(jerk_paths_ppo, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of N-PPO and C-PPO")

# Agent优劣
deviation_paths_c_sac_ppo = ['csvfile/final_deviation8_sac_final.csv','csvfile/final_deviation1_ppo_final.csv']
deviation_paths_c_sac_pp = ['csvfile/deviation0_pp2.0.csv', 'csvfile/final_deviation8_sac_final.csv']
deviation_paths_c_ppo_pp = ['csvfile/deviation0_pp2.0.csv', 'csvfile/final_deviation1_ppo_final.csv']
# draw_aligned_compare_image(deviation_paths_c_sac_ppo, "Tracking Completion Rate", "Deviation", "(m)", "Deviation Comparison of C-PPO and C-SAC")
# draw_aligned_compare_image(deviation_paths_c_sac_pp, "Tracking Completion Rate", "Deviation", "(m)", "Deviation Comparison of PP and C-SAC")
# draw_aligned_compare_image(deviation_paths_c_ppo_pp, "Tracking Completion Rate", "Deviation", "(m)", "Deviation Comparison of PP and C-PPO")

accel_paths_c_sac_ppo = ['csvfile/final_accel0_sac_final.csv','csvfile/final_accel1_ppo_final.csv']
accel_paths_c_sac_pp = ['csvfile/final_accel4_sac_final.csv','csvfile/accel3_pp2.0.csv']
accel_paths_c_ppo_pp = ['csvfile/final_accel0_ppo_final.csv','csvfile/accel3_pp2.0.csv']
# draw_aligned_compare_image(accel_paths_c_sac_ppo, "Tracking Completion Rate", "Lat-Acceleration", "(m/s^2)", "Lateral Acceleration Comparison of C-PPO and C-SAC")
# draw_aligned_compare_image(accel_paths_c_sac_pp, "Tracking Completion Rate", "Lat-Acceleration", "(m/s^2)", "Lateral Acceleration Comparison of PP and C-SAC")
# draw_aligned_compare_image(accel_paths_c_ppo_pp, "Tracking Completion Rate", "Lat-Acceleration", "(m/s^2)", "Lateral Acceleration Comparison of PP and C-PPO")

jerk_paths_c_sac_pp = ['csvfile/final_latjerk5_sac_final.csv','csvfile/latjerk0_pp2.0.csv']
jerk_paths_c_ppo_pp = ['csvfile/final_latjerk3_ppo_final.csv','csvfile/latjerk0_pp2.0.csv']
jerk_paths_c_sac_ppo = ['csvfile/final_latjerk5_sac_final.csv','csvfile/final_latjerk3_ppo_final.csv']
# draw_aligned_compare_image(jerk_paths_c_sac_pp, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of PP and C-SAC")
# draw_aligned_compare_image(jerk_paths_c_ppo_pp, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of PP and C-PPO")
# draw_aligned_compare_image(jerk_paths_c_sac_ppo, "Tracking Completion Rate", "Lat-Jerk", "(m/s^3)", "Lateral Jerk Comparison of C-PPO and C-SAC")
