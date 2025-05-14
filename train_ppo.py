import argparse
from parameters import *
import torch
import csv
import os
from PPO import PPOContinuous
from env import CarEnv
from train import train_on_policy_agent
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from visualization import RLTrainingVisualizer
import threading

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




actor_lr = PPO_ACTOR_LR
critic_lr = PPO_CRITIC_LR
num_episodes = NUM_EPISODES
hidden_dim = HIDDEN_DIM 
gamma = GAMMA # 折扣率
lmbda = LMBDA
epochs = EPOCHS
eps = EPS
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

shutdown = args['shutdown']
init_checkpoint = args['checkpoint']


env = CarEnv(render=True, sync=True, town=TOWN, pathfile=PATHFILE)
state_dim = STATE_DIM
action_dim = ACTION_DIM
data_queue = Queue()
visualizer = RLTrainingVisualizer(data_queue, args)
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
try:
    producer_thread = threading.Thread(target=train_on_policy_agent, args=(env, agent, num_episodes, args, data_queue))
    producer_thread.start()
    visualizer.start()
    return_list = data_queue['return_list']
    if args['save']:
        save_return_log(return_list, args['save'])

finally:
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format('carla'))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format('carla'))
    plt.show()
    env.destroy_env()
    if shutdown:
        os.system('shutdown /s /t 60')
    