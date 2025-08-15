from env import CarEnv
from DDPG import DDPGAgent, ReplayBuffer, PolicyNet, QValueNet
from SAC import SACContinuous, PolicyNetContinuous
from PPO import PPOPolicyNetContinuous
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import torch
from parameters import *
import argparse
import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
'''
t3_long2_r0d2_100
Average reward:1494.5200259252972
Average_deviation:0.0636992246331204
Average_speed:5.183605977052961
Average_latacc:0.6661644958262752

ckpt:t3_long2_r0d2_conv_reward_100
Average reward:-4038.394671429873
Average_deviation:0.08315828705682486
Average_speed:6.468599419176072
Average_latacc:0.9576826522079465

'''
print(f"Running script: {sys.argv[0]}")
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', type=str, required=True, help='Please enter the filename of checkpoint.')
args = vars(ap.parse_args())

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
        plt.savefig('image/'+filename, dpi=dpi)
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
        plt.savefig('image/'+filename)
    plt.show()

def evaluate_agent(env, pp_actor,obs_actor, plot=False, save=False):
    current_mode = "pp"
    total_deviation = list()
    total_speed = list()
    total_latacc = list()
    episode_reward = list()
    num_episodes = 10  # 验证10个章节
    path = list()
    for _ in range(num_episodes):
        total_reward = 0
        done = False
        speed = list()
        dist_from_center = list()
        lat_jerk = list()
        lkd = list()
        vpath = list()
        accel = list()
        rotation_v = list()
        state, path = env.reset()
 
        while not done:
            with torch.no_grad():
                pp_state = state[2:]
                obs_state = state [:6]
                ##state_tensor = torch.tensor([state], dtype=torch.float).to(device)
                pp_state_tensor = torch.tensor(np.array([pp_state]), dtype=torch.float32).unsqueeze(0).to(device)
                obs_state_tensor = torch.tensor(np.array([obs_state]), dtype=torch.float32).unsqueeze(0).to(device)
                #if env.avoidance_mode:
                mode = "obs" if env.avoidance_mode else "pp"    # 如果当前模式与上次不同，跳过该帧预测，防止模型不匹配
                if mode != current_mode:
                    print(f"[切换] 模式从 {current_mode} -> {mode}，跳过本帧避免模型不一致")
                    current_mode = mode
                    # 用 dummy action 执行一步，保持环境进展
                    _, _, done, info = env.step(np.zeros((1,))) # 注意这里改成 dummy
                    continue
                if current_mode == "obs":
                    action, log_prob = obs_actor(obs_state_tensor)
                    print(f"[INFO] 当前模式：{'RL-避障' if env.avoidance_mode else 'PP-路径跟踪'}，action = {action}")
                else:
                    action, log_prob = pp_actor(pp_state_tensor)
                    print(f"[INFO] 当前模式：{'RL-避障' if env.avoidance_mode else 'PP-路径跟踪'}，action = {action}")
            next_state, reward, done, info = env.step(action)
            current_mode= info["mode"]
            total_reward += reward
            state = next_state
            # print(f"角速度:{info[6]}速度: {int(info[0])}, 前视距离:{info[2]}, 油门: {info[1]}, Ave_devi:{info[8]}, Devi:{info[9]}")
            speed.append(info["speed"]/3.6)
            dist_from_center.append(info["deviation"])
            lat_jerk.append(info["lat_jerk"])
            vpath.append(info["vehicle_location"])
            # ave_devi.append(info[8])
            #lkd.append(info["lookahead"])
            accel.append(info["lat_accel"])
            rotation_v.append(info["v_yaw"])
        total_deviation.append(np.mean(dist_from_center))
        total_speed.append(np.mean(speed)) 
        total_latacc.append(np.mean(np.abs(accel)))
        episode_reward.append(total_reward)
        if plot and save:
            plot_array(np.array(moving_average(speed, window_size=9)), linewidth=1, filename=f'speed{_}_{args["checkpoint"]}.png', ylabel='Speed(m/s)')
            plot_array(np.array(moving_average(dist_from_center, window_size=9)), linewidth=1, filename=f'dist_from_center{_}_{args["checkpoint"]}.png', ylabel='Deviation(m)')
            plot_array(np.array(moving_average(lat_jerk, window_size=9)), linewidth=1, filename=f'lat_jerk{_}_{args["checkpoint"]}.png', ylabel='Lateral jerk(m/s^3)')
            plot_array(np.array(moving_average(lkd, window_size=9)), linewidth=1, filename=f'lookahead{_}_{args["checkpoint"]}.png', ylabel='Look-ahead(m)')
            plot_array(np.array(moving_average(accel, window_size=9)), linewidth=1, filename=f'lat_acceleration{_}_{args["checkpoint"]}.png', ylabel='Lateral Acceleration(m/s^2)')
            plot_array(np.array(moving_average(rotation_v, window_size=9)), linewidth=1, filename=f'rotationV{_}_{args["checkpoint"]}.png', ylabel='w(rad/s)')
            plot_path(np.array(path), np.array(vpath), filename=f'path{_}.png')
            # plot_array(np.array(ave_devi), markersize=1, filename=f'ave_devi{_}.png')
        elif plot:
            plot_array(np.array(moving_average(speed, window_size=9)), linewidth=1, ylabel='Speed')
            plot_array(np.array(moving_average(dist_from_center, window_size=9)), linewidth=1, ylabel='Deviation')
            plot_array(np.array(moving_average(lat_jerk, window_size=9)), linewidth=1, ylabel='Lateral jerk')
            plot_array(np.array(moving_average(lkd, window_size=9)), linewidth=1, ylabel='Look-ahead')
            plot_array(np.array(moving_average(accel, window_size=9)), linewidth=1, ylabel='Acceleration')
            plot_array(np.array(moving_average(rotation_v, window_size=9)), linewidth=1, filename=f'rotationV{_}.png', ylabel='w(rad/s)')
            plot_path(np.array(path), np.array(vpath))
        else: 
            pass
        print(f"Max speed{_}:{max(speed)}\nMax lat_acc{_}:{max(accel)}\nMax Deviation{_}:{max(dist_from_center)}")       
        save_list_to_csv(speed, f"csvfile/final_speed{_}_{args['checkpoint']}.csv")
        save_list_to_csv(dist_from_center, f"csvfile/final_deviation{_}_{args['checkpoint']}.csv")
        save_list_to_csv(lkd, f"csvfile/final_lkd{_}_{args['checkpoint']}.csv")
        save_list_to_csv(accel, f"csvfile/final_accel{_}_{args['checkpoint']}.csv")
        save_list_to_csv(lat_jerk, f"csvfile/final_latjerk{_}_{args['checkpoint']}.csv")
        save_list_to_csv(vpath, f"csvfile/final_vpath{_}_{args['checkpoint']}.csv")
    average_reward = sum(episode_reward) / num_episodes
    data_analysis={
        "mean_deviation": total_deviation,
        "mean_speed": total_speed,
        "mean_latacc": total_latacc,
        "episode_reward": episode_reward
    }
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data_analysis.items() ]))
    save_list_to_csv(path, f"Ground_truth_path.csv")
    df.to_csv(f'csvfile/t3_long2_performance_each_episode_{args["checkpoint"]}.csv', index=False)
    average_deviation = sum(total_deviation) / num_episodes
    average_speed = sum(total_speed) / num_episodes
    average_latacc = sum(total_latacc) / num_episodes
    return average_reward, average_deviation, average_speed, average_latacc
try:
    # 存储权重的文件夹名称
    saved_weights_dir = SAVED_CHECKPOINT_DIR
    ##checkpoint_file = args['checkpoint']
    # 解析两个 checkpoint 路径
    pp_checkpoint_file = args['checkpoint'] + "_pp"
    obstacle_checkpoint_file = args['checkpoint'] + "_obs"

    # 加载模型权重
    pp_actor_dict = torch.load(os.path.join(saved_weights_dir, f'{pp_checkpoint_file}.pt'))
    obstacle_actor_dict = torch.load(os.path.join(saved_weights_dir, f'{obstacle_checkpoint_file}.pt'))
    
    # 载入保存的权重
    #actor_dict = torch.load(os.path.join(saved_weights_dir, f'{checkpoint_file}_actor.pt'))

    # 创建环境和agent
    env = CarEnv(sync=True, train=False, town=TOWN, pathfile=PATHFILE)

    # 动作空间和状态空间由环境决定
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    pp_action_dim = PP_ACTION_DIM
    pp_state_dim = 5
    obs_state_dim = 6

    # 隐藏层数量
    hidden_dim = HIDDEN_DIM



    # 设备信息
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # # 创建Agent并载入权重
    # if 'ppo' in args['checkpoint']:
    #     actor = PPOPolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
    # else:
    #     actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, 1).to(device)
    # actor.load_state_dict(actor_dict)
    # # 测试模式 evaluate mode，神经网络将不会执行诸如Dropout之类的操作，而只进行inference
    # actor.eval()

    # 创建模型结构
    pp_actor = PolicyNetContinuous(pp_state_dim, hidden_dim, pp_action_dim, 1).to(device)
    obstacle_actor = PolicyNetContinuous(obs_state_dim, hidden_dim, action_dim, 1).to(device)
    # 加载权重
    pp_actor.load_state_dict(pp_actor_dict)
    obstacle_actor.load_state_dict(obstacle_actor_dict)
    # 设置为评估模式
    pp_actor.eval()
    obstacle_actor.eval()



    # Evaluate the performance of the agent using the loaded Q-network
    average_reward, average_deviation, average_speed, average_latacc = evaluate_agent(env, pp_actor, obstacle_actor)
    print(f"[pp_model]\nAverage reward:{average_reward}\nAverage_deviation:{average_deviation}\nAverage_speed:{average_speed}\nAverage_latacc:{average_latacc}")
    print(f"[Obstacle_model]\nAverage reward:{average_reward}\nAverage_deviation:{average_deviation}\nAverage_speed:{average_speed}\nAverage_latacc:{average_latacc}")
finally:
    env.destroy_env()
