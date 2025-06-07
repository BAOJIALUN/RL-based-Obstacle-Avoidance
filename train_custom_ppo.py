
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
import faulthandler
faulthandler.enable()

def save_return_log(return_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Return'])
        for i, episode_return in enumerate(return_list):
            writer.writerow([i + 1, episode_return])
            
            
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", required=False, help="save the weights as input name", default=None)
    ap.add_argument("--shutdown", required=False, help="shutdown the computer after finished training", default=False)
    ap.add_argument("--checkpoint", required=False, help="Initialize from checkpoint", default=None)
    args = vars(ap.parse_args())    # 初始化
    save_dir = SAVED_CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    env = CarEnv(render=True, sync=True, town=TOWN, pathfile=PATHFILE)
    agent = PPOContinuous(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        actor_lr=PPO_ACTOR_LR,
        critic_lr=PPO_CRITIC_LR,
        lmbda=LMBDA,
        epochs=EPOCHS,
        eps=EPS,
        gamma=GAMMA,
        device=device,
    )    # 训练
    return_list = train_on_policy_agent(env, agent, NUM_EPISODES, args)    
    
    if args['save']:
        save_return_log(return_list, args['save'])    # 可视化
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on Carla')
    plt.show()    
    
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Smoothed PPO Returns')
    plt.show()    
    
    env.destroy_env()    
    
    if args['shutdown']:
        os.system('shutdown /s /t 60')


