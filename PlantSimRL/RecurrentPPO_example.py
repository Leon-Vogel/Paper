import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
model.learn(5000)

env = model.get_env()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
print(mean_reward)

model.save("ppo_recurrent")
del model  # remove to demonstrate saving and loading

model = RecurrentPPO.load("ppo_recurrent")

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()


'''
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
import os

# Initialisiere das Environment
env_id = 'CartPole-v0'
env = gym.make(env_id)
env = Monitor(env, log_dir, allow_early_resets=True)

# Erstelle ein Vectorized Environment
env = DummyVecEnv([lambda: env])

# Initialisiere das Modell
model = PPO('MlpPolicy', env, tensorboard_log=log_dir)

# Initialisiere Callbacks
eval_callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000, n_eval_episodes=5)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

# Trainiere das Modell
model.learn(total_timesteps=100000, callback=[eval_callback, stop_callback])

# Visualisiere die Trainingsergebnisse mit Tensorboard
results = load_results(log_dir)
x, y = ts2xy(results, 'timesteps')
logger.configure(log_dir)
logger.record('timesteps', x[-1])
logger.record('mean_reward', y[-1])
logger.dumpkvs()
os.system(f"tensorboard --logdir {log_dir} --port 6006")

'''