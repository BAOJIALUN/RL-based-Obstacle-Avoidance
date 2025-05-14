import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from env import CarEnv
# ← 确保你这个类在 env.py 里定义好了# 
# ==== 日志输出目录 ====
logdir = f"logs/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
# ==== 创建环境 ====
env = CarlaPPObstacleEnv()  # 你可以加上参数，例如 CarlaPPObstacleEnv(args)
env = Monitor(env, logdir)  # 加 monitor 能记录日志供 tensorboard 查看
# ==== 创建模型 ====
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    tensorboard_log=logdir,
)
# ==== 回调函数（保存模型） ====
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=logdir,
    name_prefix="ppo_checkpoint"
)
# ==== 启动训练 ====
model.learn(
    total_timesteps=100_000,
    callback=CallbackList([checkpoint_callback]),
    tb_log_name="ppo_mlp",
    progress_bar=True,
)
# ==== 保存模型 ====
model_path = os.path.join(logdir, "ppo_obstacle_mlp")
model.save(model_path)
print(f":白色的对勾: 模型保存到: {model_path}")