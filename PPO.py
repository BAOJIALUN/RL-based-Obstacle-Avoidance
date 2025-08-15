'''
Description: Agent.  
Modified by: Pang.
License: Apache-2.0 license
Original Author: Boyu-ai Group
Original project URL: https://github.com/boyu-ai/Hands-on-RL
'''




import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import os
from parameters import*
def set_seed(seed):
    '''
    设置随机种子以初始化权重
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_advantage(gamma, lmbda, td_delta):
    '计算优势函数'
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PPOPolicyNetContinuous(torch.nn.Module):
    '连续策略网络'
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPOPolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

        # :白色的对勾: 加上权重初始化（防止 fc1 输出爆炸）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_std.weight)
        nn.init.zeros_(self.fc_std.bias)

    # def forward(self, x):
    #     print("[DEBUG] input x:", x)    # 限幅输入，避免爆炸
    #     x = torch.clamp(x, -10.0, 10.0)
    #     x = F.relu(self.fc1(x))    # mu 输出使用 tanh 限制在 [-1, 1] 区间
    #     mu = torch.tanh(self.fc_mu(x))    # std 使用 softplus 保证正值，再加 clamp 限制下界
    #     std = F.softplus(self.fc_std(x))
    #     std = torch.clamp(std, min=1e-3, max=1.0)    
    #     # --- Debug：检测是否出现非法值 ---
    #     if torch.isnan(mu).any() or torch.isinf(mu).any():
    #         print("[ERROR] :x: mu 出现非法值:", mu)
    #     if torch.isnan(std).any() or torch.isinf(std).any():
    #         print("[ERROR] :x: std 出现非法值:", std)    
    #     return mu, std
    
    def forward(self, x):
        x = torch.clamp(x, -10.0, 10.0)
        x = F.relu(self.fc1(x))
        raw_mu = self.fc_mu(x)
        raw_std = self.fc_std(x)  
        if torch.isnan(raw_mu).any() or torch.isnan(raw_std).any():
            print("[ERROR] :x: raw_mu 或 raw_std 出现 NaN！")
            raise ValueError("actor raw output has nan")    
        mu = torch.tanh(raw_mu)
        std = F.softplus(raw_std)
        std = torch.clamp(std, min=1e-3, max=1.0)    
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("[ERROR] :x: mu/std 出现 NaN！")
            raise ValueError("actor final output has nan")    
        return mu, std
        
    # def forward(self, x):
    #     print("[DEBUG] 输入 x:", x)    
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print(":x: 输入状态有 NaN 或 Inf")
    #         raise ValueError("非法输入")    
    #     x = F.relu(self.fc1(x))
    #     if torch.isnan(x).any():
    #         print(":x: fc1 输出含 NaN")
    #         raise ValueError("fc1 出错")    
    #     mu = torch.tanh(self.fc_mu(x))
    #     std = F.softplus(self.fc_std(x)) + 1e-6    
    #     if torch.isnan(mu).any() or torch.isnan(std).any():
    #         print(":x: mu 或 std 中出现 NaN")
    #         raise ValueError("输出出错")    F
    #     return mu, std

class ValueNet(torch.nn.Module):
    '价值网络'
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, seed=3, train=True):
        
        if seed is not None:
            set_seed(seed)
        self.actor = PPOPolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.train = train

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.cpu().numpy()
        #return action.cpu().numpy()


    def update(self, transition_dict):
        
        def safe_array(arr,name=""):
            arr = np.array(arr, dtype=np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"[SKIP WARNING] {name} 含有 NaN 或 Inf，跳过本轮更新")
                return None
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  
        # --- 安全预处理，每一步都检查 ---
            
        for key in ['states', 'actions', 'rewards', 'next_states', 'dones']:
                arr = safe_array(transition_dict[key], key)
                if arr is None:
                    return # 跳过当前 update
                transition_dict[key] = arr
        # 安全预处理 transition_dict 的内容
        transition_dict['states'] = safe_array(transition_dict['states'])
        transition_dict['next_states'] = safe_array(transition_dict['next_states'])
        transition_dict['actions'] = safe_array(transition_dict['actions'])
        transition_dict['rewards'] = safe_array(transition_dict['rewards'])
        transition_dict['dones'] = safe_array(transition_dict['dones'])
        

        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               #dtype=torch.float).view(-1, 2).to(self.device)
                               #dtype=torch.float).view(-1, 1).to(self.device)
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("[ERROR] actor 输出含 NaN，本次 update 跳过")
            return
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def load_checkpoint(self, checkpoint):
        actor_dict = torch.load(os.path.join(SAVED_CHECKPOINT_DIR, f'{checkpoint}_actor.pt'))
        if self.train:
            critic_dict = torch.load(os.path.join(SAVED_CHECKPOINT_DIR, f'{checkpoint}_critic.pt'))
            self.critic.load_state_dict(critic_dict)

        self.actor.load_state_dict(actor_dict)
        print("checkpoint loaded")
