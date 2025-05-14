from env import CarEnv
from parameters import*
from PID import PIDController

import numpy as np

import matplotlib.pyplot as plt
import csv
import pandas as pd

def save_list_to_csv(data, filename):
    """
    保存单一列表为CSV文件

    参数:
    data (list): 要保存的列表
    filename (str): 文件名
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def load_list_from_csv(filename):
    """
    从CSV文件恢复单一列表

    参数:
    filename (str): 文件名

    返回:
    list: 恢复的列表
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data_from_csv = next(reader)  # 读取一行
        return [int(i) for i in data_from_csv]  # 将字符串转换为整数

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
    
def plot_array(arr, linewidth=2, filename=None, xlabel='Time step', ylabel=None, fontsize=14, dpi=300):
    if arr.ndim == 1:
        plt.plot(np.arange(len(arr)), arr, linewidth=linewidth)
    elif arr.ndim == 2:
        for i, row in enumerate(arr):
            plt.plot(np.arange(len(row)), row, linewidth=linewidth, label=f'Line {i}')
        plt.legend()
    else:
        print("Unsupported array dimension. Only 1D or 2D arrays are supported.")
        return
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    if filename:
        plt.savefig(filename, dpi=dpi)
    plt.show()
def plot_path(path1, path2, filename=None):
    '''
    画出路径和车辆跟踪的路径
    '''
    array1_x = [point[0] for point in path1]
    array1_y = [point[1] for point in path1]

    array2_x = [point[0] for point in path2]
    array2_y = [point[1] for point in path2]

    # 绘制图形
    plt.scatter(array1_x, array1_y, color='green', label='path', s=3)
    plt.scatter(array2_x, array2_y, color='red', label='vpath', s=3)
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def evaluate_agent(env, plot=False, save=True):
    episode_reward = list()
    total_deviation = list()
    total_speed = list()
    total_latacc = list()
    num_episodes = 10  # 验证10个章节
    for _ in range(num_episodes):
        total_reward = 0
        done = False
        speed = list()
        dist_from_center = list()
        lat_jerk = list()
        lkd = list()
        vpath = list()
        path = list() 
        accel = list()
        rotation_v = list()
        state, path = env.reset()
 
        while not done:

            action = [[-1.0, 0.32]]
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            # print(f"角速度:{info[6]}速度: {int(info[0])}, 前视距离:{info[2]}, 油门: {info[1]}, Ave_devi:{info[8]}, Devi:{info[9]}")
            speed.append(info['speed']/3.6)
            dist_from_center.append(info['deviation'])
            lat_jerk.append(info['lat_jerk'])
            vpath.append(info['vehicle_location'])
            lkd.append(info['lookahead'])
            accel.append(info['lat_accel'])
            rotation_v.append(info['v_yaw'])
        total_deviation.append(np.mean(dist_from_center))
        total_speed.append(np.mean(speed)) 
        total_latacc.append(np.mean(np.abs(accel)))
        episode_reward.append(total_reward)
        if plot and save:
            plot_array(np.array(moving_average(speed, window_size=9)), linewidth=1, filename=f'PIDspeed{_}.png', ylabel='Speed(m/s)')
            plot_array(np.array(moving_average(dist_from_center, window_size=9)), linewidth=1, filename=f'PIDdist_from_center{_}.png', ylabel='Deviation(m)')
            plot_array(np.array(moving_average(lat_jerk, window_size=9)), linewidth=1, filename=f'PIDlat_jerk{_}.png', ylabel='Lateral jerk(m/s^3)')
            plot_array(np.array(moving_average(lkd, window_size=9)), linewidth=1, filename=f'PIDlookahead{_}.png', ylabel='Look-ahead(m)')
            plot_array(np.array(moving_average(accel, window_size=9)), linewidth=1, filename=f'PIDlat_acceleration{_}.png', ylabel='Lateral Acceleration(m/s^2)')
            plot_array(np.array(moving_average(rotation_v, window_size=9)), linewidth=1, filename=f'PIDrotationV{_}.png', ylabel='w(rad/s)')
            plot_path(np.array(path), np.array(vpath), filename=f'PIDpath{_}.png')
            # plot_array(np.array(ave_devi), markersize=1, filename=f'ave_devi{_}.png')
        elif plot:
            plot_array(np.array(speed), linewidth=1, ylabel='Speed')
            plot_array(np.array(dist_from_center), linewidth=1, ylabel='Deviation')
            plot_array(np.array(lat_jerk), linewidth=1, ylabel='Lateral jerk')
            plot_array(np.array(lkd), linewidth=1, ylabel='Look-ahead')
            plot_array(np.array(accel), linewidth=1, ylabel='Acceleration')
            plot_array(np.array(rotation_v), linewidth=1, filename=f'rotationV{_}.png', ylabel='w(rad/s)')
            plot_path(np.array(path), np.array(vpath))
        else: 
            pass
        print(f"Max speed{_}:{max(speed)}\nMax lat_acc{_}:{max(accel)}\nMax Deviation{_}:{max(dist_from_center)}")    
        save_list_to_csv(speed, f"csvfile/speed{_}_pid.csv")
        save_list_to_csv(dist_from_center, f"csvfile/deviation{_}_pid.csv")
        save_list_to_csv(accel, f"csvfile/accel{_}_pid.csv")
        save_list_to_csv(lat_jerk, f"csvfile/latjerk{_}_pid.csv")
        save_list_to_csv(vpath, f"csvfile/vpath{_}_pid.csv")
    
    data_analysis={
        "mean_deviation": total_deviation,
        "mean_speed": total_speed,
        "mean_latacc": total_latacc,
        "episode_reward": episode_reward
    }
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data_analysis.items() ]))
    df.to_csv(f't3_right_performance_each_episode_pid.csv', index=False)
    average_reward = sum(episode_reward) / num_episodes
    average_deviation = sum(total_deviation) / num_episodes
    average_speed = sum(total_speed) / num_episodes
    average_latacc = sum(total_latacc) / num_episodes
    return average_reward, average_deviation, average_speed, average_latacc


try:


    # 创建环境
    env = CarEnv(sync=True, train=False, town=TOWN, pathfile=PATHFILE, pid=PIDController(PID_KP, PID_KI, PID_KD, 0.05))

    # Evaluate the performance of the agent using the loaded Q-network
    average_reward, average_deviation, average_speed, average_latacc = evaluate_agent(env)
    print(f"ckpt:{'no checkpoint'}\nAverage reward:{average_reward}\nAverage_deviation:{average_deviation}\nAverage_speed:{average_speed}\nAverage_latacc:{average_latacc}")
finally:
    env.destroy_env()