"""
Description: Trainning functions of on-policy and off-policy   
Modified by: Pang.
License: Apache-2.0 license
Original Author: Boyu-ai Group
Original project URL: https://github.com/boyu-ai/Hands-on-RL

added functional: Save the highest rewarded model
"""

from visualization import RLTrainingVisualizer
from tqdm import tqdm
from parameters import *
import torch
import numpy as np
import argparse
import os
import csv
import threading
def run_visualizer(visualizer):
    visualizer.start()
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, args, queue):
    return_list = []
    best_retrun = -np.inf
    best_episode = 0
    # 10轮
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            # 每轮num/10个episode
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                    new_data = {
                        "reward": info["reward"],
                        "speed":info["speed"],
                        "lookahead":info["lookahead"],
                        "throttle":info["throttle"],
                        "deviation":info["deviation"]
                    }
                    queue.put(new_data)
                return_list.append(episode_return)
                queue.put({"reset":True})
                print(f"Epsiode reward:{int(episode_return)}\n time:{info['timesteps']}\n lat_jerk:{info['lat_accel']}\n cover:{info['path_covered']}" )
                if episode_return > best_retrun:
                    best_retrun = episode_return
                    best_episode = i * int(num_episodes/10) + i_episode + 1
                    if args['save']:
                        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_actor.pt"))
                        torch.save(agent.critic_1.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic1.pt"))
                        torch.save(agent.critic_2.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic2.pt"))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    if args['save']:
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{num_episodes}_actor.pt"))
        torch.save(agent.critic_1.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic1.pt"))
        torch.save(agent.critic_2.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic2.pt"))
    queue.put({"return_list": return_list})

"""
def train_on_policy_agent(env, agent, num_episodes, args, queue):
    return_list = []
    best_retrun = -np.inf
    best_episode = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, tracking_path = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    new_data = {
                        # "reward": info["reward"],
                        # "speed":info["speed"],
                        # "lookahead":info["lookahead"],
                        # "throttle":info["throttle"],
                        # "deviation":info["deviation"]
                        "reward": info.get("reward", 0),
                        "speed": info.get("speed", 0),
                        "throttle": info.get("throttle", 0),
                        "steer": info.get("steer", 0),
                        "angle": info.get("angle", 0),
                        "deviation": info.get("distance_from_center", 0),
                        "collision": info.get("collision", 0),
                        "lane": info.get("lane", 0),
                        "timesteps": info.get("timesteps", 0)
                    }
                    queue.put(new_data)

                return_list.append(episode_return)
                queue.put({"reset":True})
                print(f"Epsiode reward:{int(episode_return)}\n time:{info['timesteps']}\n lat_jerk:{info['lat_accel']}\n cover:{info['path_covered']}" )
                agent.update(transition_dict)
                if episode_return > best_retrun:
                    best_retrun = episode_return
                    best_episode = i * int(num_episodes/10) + i_episode + 1
                    if args['save']:
                        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_actor.pt"))
                        torch.save(agent.critic.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic.pt"))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    if args['save']:
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{num_episodes}_actor.pt"))
        torch.save(agent.critic.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic.pt"))
    save_return_log(return_list)
    queue.put({"return_list": return_list})
"""

def train_on_policy_agent(env, agent, num_episodes, args, queue=None):
    return_list = []
    best_return = -np.inf
    best_episode = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward

                    if queue is not None:
                        new_data = {
                            "reward": info.get("reward", 0),
                            "speed": info.get("speed", 0),
                            "throttle": info.get("throttle", 0),
                            "steer": info.get("steer", 0),
                            "angle": info.get("angle", 0),
                            "deviation": info.get("distance_from_center", 0),
                            "collision": info.get("collision", 0),
                            "lane": info.get("lane", 0),
                            "timesteps": info.get("timesteps", 0)
                        }
                        queue.put(new_data)

                return_list.append(episode_return)

                if queue is not None:
                    queue.put({"reset": True})

                print(f"Episode reward: {int(episode_return)}")
                agent.update(transition_dict)

                if episode_return > best_return:
                    best_return = episode_return
                    best_episode = i * int(num_episodes / 10) + i_episode + 1
                    if args['save']:
                        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_actor.pt"))
                        torch.save(agent.critic.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic.pt"))

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': f'{num_episodes / 10 * i + i_episode + 1}', 'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)

    if args['save']:
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"{args['save']}_{num_episodes}_actor.pt"))
        torch.save(agent.critic.state_dict(), os.path.join(save_dir, f"{args['save']}_{best_episode}_critic.pt"))

    save_return_log(return_list)

    if queue is not None:
        queue.put({"return_list": return_list})

    return return_list



def save_return_log(return_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Return'])
        for i, episode_return in enumerate(return_list):
            writer.writerow([i+1, episode_return])

save_dir = SAVED_CHECKPOINT_DIR
os.makedirs(save_dir, exist_ok=True) 

