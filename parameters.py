
# 权重保存路径
SAVED_CHECKPOINT_DIR = 'checkpoints'

# DDPG
ACTOR_LR = 3e-4 # actor 学习率
CRITIC_LR = 3e-3 # critic 学习率
NUM_EPISODES = 350 # 训练总episode
HIDDEN_DIM = 128 # 隐藏层大小
GAMMA = 0.99 # 折扣率
TAU = 0.01  # 软更新参数
BUFFER_SIZE = 100000 # 经验回放池大小
MINIMAL_BUFFER_SIZE = 1000 # 经验回放池最小size
BATCH_SIZE = 64 # 单次更新时的batch size
SIGMA = 0.05  # 高斯噪声标准差
# SAC
ALPHA_LR = 3e-4

# PPO
PPO_ACTOR_LR = 3e-4 # actor 学习率
PPO_CRITIC_LR = 3e-3 # critic 学习率
LMBDA = 0.95
EPOCHS = 20
EPS = 0.2
## carlaenv 
HOST = '127.0.0.1'
PORT = 2000
IM_WIDTH = 640
IM_HEIGHT = 480
MAX_DISTANCE_FROM_CENTER = 3
STATE_DIM = 6
ACTION_DIM = 1
TOWN = 'Town05'
PATHFILE = 'town5_waypoints2'
# TOWN = 'Town03'
# PATHFILE = 'town3_long2_r0d2'

## High speed
H_MAX_SPEED = 40
H_TARGET_SPEED = 35
H_MIN_SPEED = 30

## Mid speed
M_MAX_SPEED = 35
M_TARGET_SPEED = 30
M_MIN_SPEED = 25

## Low speed
L_MAX_SPEED = 30
L_TARGET_SPEED = 25
L_MIN_SPEED = 20

## PIDController Parameters
PID_KP = .5
PID_KI = .25
PID_KD = .1