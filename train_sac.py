from parameters import *
from SAC import SACContinuous, ReplayBuffer
from env import CarEnv
from train import train_off_policy_agent

import threading
import argparse
import torch
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from visualization import RLTrainingVisualizer

def save_return_log(return_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Return'])
        for i, episode_return in enumerate(return_list):
            writer.writerow([i+1, episode_return])
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

ap = argparse.ArgumentParser()
ap.add_argument("--save", required=False,
	help="save the weights as input name", default=None)
ap.add_argument("--shutdown", required=False,
	help="shutdown the computer after finished training", default=False)
ap.add_argument("--checkpoint", required=False,
	help="Initialize the actor and critic by input checkpoint", default=None)
args = vars(ap.parse_args())


save_dir = SAVED_CHECKPOINT_DIR
os.makedirs(save_dir, exist_ok=True) 




actor_lr = ACTOR_LR
critic_lr = CRITIC_LR
num_episodes = NUM_EPISODES
hidden_dim = HIDDEN_DIM 
gamma = GAMMA # 折扣率
tau = TAU  # 软更新参数
buffer_size = BUFFER_SIZE # 最大经验回放池大小
minimal_size = MINIMAL_BUFFER_SIZE # 最小经验回放池大小
batch_size = BATCH_SIZE # 单次更新时Batch Size
alpha_lr = ALPHA_LR
target_entropy = -ACTION_DIM
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#数据可视化用的通讯队列
data_queue = Queue()
visualizer = RLTrainingVisualizer(data_queue, args)

shutdown = args['shutdown']
init_checkpoint = args['checkpoint']

replay_buffer = ReplayBuffer(buffer_size)
env = CarEnv(render=True, sync=True, town=TOWN, pathfile=PATHFILE)

state_dim = env.state_dim
action_dim = env.action_dim
action_bound = 1 # 输出动作按原样输出。lookahead的输出上限会在env中给出
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, 
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, checkpoint=init_checkpoint)

try:
    producer_thread = threading.Thread(target=train_off_policy_agent, args=(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, args, data_queue))
    producer_thread.start()
    visualizer.start()

finally:
    data = data_queue.get()
    return_list=data['return_list']
    if args['save']:
        save_return_log(return_list, args['save'])
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format('carla'))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format('carla'))
    plt.show()
    env.destroy_env()
    if shutdown:
        os.system('shutdown /s /t 60')
    