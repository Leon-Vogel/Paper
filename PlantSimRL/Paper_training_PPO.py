from time import sleep

from ps_environment_sb3 import Environment
import numpy as np
from plantsim.plantsim import Plantsim
from agents.ppo_torch import PPOAgent
from utils import plot_learning_curve
import os
import torch as T
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\PPO_sb3'
simulation = pfad + '\RL_Sim_20230227.spp'

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=simulation,
                        socket=None, visible=True)
    env = Environment(plantsim)
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=speicherort,
                         device=T.device('cuda:0'if T.cuda.is_available() else'cpu'))
    model.learn(500)

    ''' # Erstellen Sie ein Trainingsumgebung und überwachen Sie sie mit Monitor
    env = Monitor(DummyVecEnv([lambda: gym.make("CartPole-v1")]), "./logs", allow_early_resets=True)

    # Konfigurieren Sie das Protokollierungsverhalten
    configure(tensorboard_log)

    # Erstellen Sie ein Modell und fügen Sie das Tensorboard-Logging hinzu
    model = PPO("MlpPolicy", env, tensorboard_log=tensorboard_log)

    # Definieren Sie Callbacks, um das Training zu überwachen
    eval_callback = EvalCallback(env, eval_freq=1000, deterministic=True, render=False)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=195.0, verbose=1)

    # Trainieren Sie das Modell
    model.learn(total_timesteps=int(1e6), callback=[eval_callback, stop_callback])'''

    env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    model.save("ppo_recurrent")
    del model

    model = RecurrentPPO.load("ppo_recurrent")
    obs = env.reset()
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        env.render()

