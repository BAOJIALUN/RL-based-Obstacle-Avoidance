# reference url: https://hrl.boyuai.com/chapter/2/ddpg%E7%AE%97%E6%B3%95#132-ddpg-%E7%AE%97%E6%B3%95


import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import os
from parameters import *

#===========================
#----------网络组件----------
#===========================

'''
策略网略(Actor network)
'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

'''
价值网络(Critic network)
'''
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 64)
        self.fc_out = torch.nn.Linear(64, 1)
        self.counter = 0

    def forward(self, x, a):
        self.counter+=1
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = torch.tanh(self.fc1(cat))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)

        
        
        
    


#===========================
#--------DDPG代理代码--------
#===========================
    
class DDPGAgent:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, checkpoint=None):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.seed = 42
        torch.manual_seed(self.seed)
        if not checkpoint:
            # 初始化目标价值网络并设置和价值网络相同的参数
            self.target_critic.load_state_dict(self.critic.state_dict())
            # 初始化目标策略网络并设置和策略相同的参数
            self.target_actor.load_state_dict(self.actor.state_dict())
        elif checkpoint:
            self.load_checkpoint(checkpoint)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    '''
    动作选择，其中添加了随机噪声
    '''
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)
        # 给动作添加噪声，增加探索
        noise = torch.tensor(np.random.randn(1, self.action_dim), dtype=torch.float).to(self.device)
        action = action + self.sigma * noise
        return action.tolist()
    
    '''
    软更新策略
    '''
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    '''
    更新目标网略
    '''
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 2).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def load_checkpoint(self, checkpoint):
        actor_dict = torch.load(os.path.join(SAVED_CHECKPOINT_DIR, f'{checkpoint}_actor.pt'))
        critic_dict = torch.load(os.path.join(SAVED_CHECKPOINT_DIR, f'{checkpoint}_critic.pt'))
        self.target_critic.load_state_dict(critic_dict)
        self.target_actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)
        self.actor.load_state_dict(actor_dict)
        print("checkpoint loaded")

class ReplayBuffer():
    '''
    经验回放池
    '''
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def size(self):
        return len(self.buffer)