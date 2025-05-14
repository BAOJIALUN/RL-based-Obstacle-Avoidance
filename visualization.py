import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time
import csv
from queue import Queue



class RLTrainingVisualizer:
    def __init__(self, data_queue, args):
        # 初始化数据存储列表
        self.data_queue = data_queue
        self.reward_data = []
        self.deviation_data = []
        self.speed_data = []
        self.lookahead_data = []
        self.throttle_data = []
        self.running = True  # 控制可视化运行状态的标志
        self.args = args

        # 创建图形对象和4个子图
        self.fig, self.axs = plt.subplots(5, 1, figsize=(8, 12))

        # 初始化每个子图的线条对象
        self.line_reward, = self.axs[0].plot([], [], 'r-', lw=2, label='Reward')
        self.line_deviation, = self.axs[1].plot([], [], 'g-', lw=2, label='Deviation')
        self.line_speed, = self.axs[2].plot([], [], 'b-', lw=2, label='Speed')
        self.line_lookahead, = self.axs[3].plot([], [], 'm-', lw=2, label='Lookahead')
        self.line_throttle, = self.axs[4].plot([], [], 'y-', lw=2, label='Throttle')

        # 分别设置每个子图的坐标轴范围
        self.axs[0].set_xlim(0, 100)
        self.axs[0].set_ylim(-1, 1)

        self.axs[1].set_xlim(0, 100)
        self.axs[1].set_ylim(0, 2)

        self.axs[2].set_xlim(0, 100)
        self.axs[2].set_ylim(0, 30)

        self.axs[3].set_xlim(0, 100)
        self.axs[3].set_ylim(0, 7)

        self.axs[4].set_xlim(0, 100)
        self.axs[4].set_ylim(-1, 1)
        # 为每个子图添加图例
        for ax in self.axs:
            ax.legend()

        # 创建动画对象
        self.ani = animation.FuncAnimation(self.fig, self.update, init_func=self.init, blit=True, interval=100)

        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def init(self):
        # 初始化所有线条数据为空
        self.line_reward.set_data([], [])
        self.line_deviation.set_data([], [])
        self.line_speed.set_data([], [])
        self.line_lookahead.set_data([], [])
        self.line_throttle.set_data([], [])
        return self.line_reward, self.line_deviation, self.line_speed, self.line_lookahead, self.line_throttle

    def update(self, frame):
        if not self.running:
            return self.line_reward, self.line_deviation, self.line_speed, self.line_lookahead, self.line_throttle

        while not self.data_queue.empty():
            data = self.data_queue.get()
            
            # 检查是否为重置信号
            if 'reset' in data and data['reset']:
                # 如果接收到重置信号，清空数据
                self.reset()
            if 'return_list' in data and data['return_list']:
                return_list=data['return_list']
                if self.args['save']:
                    self.save_return_log(return_list, self.args['save'])    
                episodes_list = list(range(len(return_list)))
                plt.plot(episodes_list, return_list)
                plt.xlabel('Episodes')
                plt.ylabel('Returns')
                plt.title('SAC on {}'.format('carla'))
                plt.show()
            else:
                # 确保数据包含所有预期的键
                if 'reward' in data and 'deviation' in data and 'speed' in data and 'lookahead' in data and 'throttle' in data:
                    # 否则，接收正常数据并更新存储列表
                    self.reward_data.append(data['reward'])
                    self.deviation_data.append(data['deviation'])
                    self.speed_data.append(data['speed'])
                    self.lookahead_data.append(data['lookahead'])
                    self.throttle_data.append(data['throttle'])

        # 更新每个子图的数据
        xdata = np.arange(len(self.reward_data))

        self.line_reward.set_data(xdata, self.reward_data)
        self.line_deviation.set_data(xdata, self.deviation_data)
        self.line_speed.set_data(xdata, self.speed_data)
        self.line_lookahead.set_data(xdata, self.lookahead_data)
        self.line_throttle.set_data(xdata, self.throttle_data)

        # 动态调整每个子图的X轴范围
        for ax in self.axs:
            ax.set_xlim(0, max(100, len(self.reward_data)))

        return self.line_reward, self.line_deviation, self.line_speed, self.line_lookahead, self.line_throttle


    def on_key_press(self, event):
        # 如果按下的是 'Esc' 键，则停止可视化
        if event.key == 'escape':
            self.running = False
            plt.close(self.fig)
            print("Visualization stopped by user.")

    def start(self):
        # 运行动画和可视化
        plt.show()
    def reset(self):
        self.reward_data = []
        self.deviation_data = []
        self.speed_data = []
        self.lookahead_data = []
        self.throttle_data = []

    def save_return_log(self, return_list, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Return'])
            for i, episode_return in enumerate(return_list):
                writer.writerow([i+1, episode_return])