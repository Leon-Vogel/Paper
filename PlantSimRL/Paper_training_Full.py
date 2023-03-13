from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment
import numpy as np
import os
import torch as T
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import functional as F
from stable_baselines3.common.env_checker import check_env
from CustomCallbacks import CustomCallback

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parameter für alle Trainings festlegen, Netz, lernrate etc.
# Directory für Ergebnisse definieren (evtl mit Dictionary?)
# pfad = 'E:\\Studium\Projekt\Paper\PlantSimRL\simulations'
# pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
erg = 'ergebnisse\\'
mod = 'models\\'
Training = {
    'Sim': [pfad + '\RL_Sim_20230310.spp', pfad + '\RL_Sim_20230313_sparse.spp', pfad + '\RL_Sim_20230310.spp',
            pfad + '\RL_Sim_20230313_sparse.spp'],
    'Logs': [erg + 'R_In_PPO', erg + 'R_Sp_PPO', erg + 'R_In_PPO_LSTM', erg + 'R_Sp_PPO_LSTM'],
    'Logname': ['256_128_64', '256_128_64', '256_128_64', '256_128_64'],
    'Model': [mod + 'R_In_PPO', mod + 'R_Sp_PPO', mod + 'R_In_PPO_LSTM', mod + 'R_Sp_PPO_LSTM']
}
'''
Training = {
    'Sim': [pfad + '\RL_Sim_20230310.spp', pfad + '\R2.spp', pfad + '\R3.spp', pfad + '\RL_Sim_20230310.spp',
            pfad + '\R2.spp', pfad + '\R3.spp'],
    'Logs': [erg + 'R1_PPO', erg + 'R2_PPO', erg + 'R3_PPO', erg + 'R1_PPO_LSTM', erg + 'R2_PPO_LSTM',
             erg + 'R3_PPO_LSTM'],
    'Model': [mod + 'R1_PPO', mod + 'R2_PPO', mod + 'R3_PPO', mod + 'R1_PPO_LSTM', mod + 'R2_PPO_LSTM',
             mod + 'R3_PPO_LSTM'],
}'''
# os.makedirs(logs, exist_ok=True)
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 256, 128, 64], vf=[256, 256, 128, 64]))
policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[128, 64], vf=[128, 64]))
eval_freq = 4100
n_eval_episodes = 3
learning_rate = 5e-4
n_epochs = 10
n_steps = 384  # 1408
clip_range = 0.15
clip_range_vf = 0.15
total_timesteps = 55000
visible = False
info_keywords = tuple(['Typ1', 'Typ2', 'Typ3', 'Typ4', 'Typ5', 'Warteschlangen', 'Auslastung'])

for i in range(2):
    os.makedirs(Training['Logs'][i], exist_ok=True)
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=visible)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i], info_keywords=info_keywords)
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i], log_path=Training['Logs'][i],
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_on_new_best=stop_callback)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=Training['Logs'][i], learning_rate=learning_rate,
                n_epochs=n_epochs, clip_range=clip_range,
                device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=clip_range_vf,
                n_steps=n_steps, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=[rollout_callback, eval_callback],
                tb_log_name=Training['Logname'][i], progress_bar=True)
    model.save(Training['Model'][i] + '\\train_model')
    del model

    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = PPO.load(Training['Model'][i] + '\\best_model', env)
    '''
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    '''
    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    env.close()

for i in range(2, 4):
    os.makedirs(Training['Logs'][i], exist_ok=True)
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=visible)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i], info_keywords=info_keywords)
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i], log_path=Training['Logs'][i],
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_on_new_best=stop_callback)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=Training['Logs'][i],
                         learning_rate=learning_rate, n_epochs=n_epochs,
                         clip_range=clip_range,
                         device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=clip_range_vf,
                         n_steps=n_steps, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=[rollout_callback, eval_callback],
                tb_log_name=Training['Logname'][i], progress_bar=True)
    model.save(Training['Model'][i] + '\\train_model')
    del model

    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = RecurrentPPO.load(Training['Model'][i] + '\\best_model', env)
    '''
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    '''
    obs = env.reset()
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    dones = False
    while not dones:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
    env.close()
