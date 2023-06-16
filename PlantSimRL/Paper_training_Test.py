import os
import random
from typing import Callable

import numpy as np
import torch as T
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from CustomCallbacks import CustomCallback
from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parameter für alle Trainings festlegen, Netz, lernrate etc.
# Directory für Ergebnisse definieren (evtl mit Dictionary?)
# pfad = 'E:\\Studium\Projekt\Paper\PlantSimRL\simulations'
# pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
erg = 'ergebnisse_Test6\\'
mod = 'models_Test3\\'  # _V1
net_arch = dict(pi=[256, 256, 128, 64], vf=[256, 256, 128, 64])
Training = {
    'Sim': [pfad + '\RL_Sim_V00_inter.spp', pfad + '\RL_Sim_V1_inter.spp',
            # pfad + '\RL_Sim_V20_inter.spp', pfad + '\RL_Sim_V3_inter.spp',
            pfad + '\RL_Sim_V4_inter.spp', pfad + '\RL_Sim_V5_inter.spp'],
    'Logs': [erg + 'R_V00_PPO', erg + 'R_V1_PPO',
             # erg + 'R_V20_PPO', erg + 'R_V3_PPO',
             erg + 'R_V4_PPO', erg + 'R_V5_PPO'],
    'Logname': [str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1',
                # str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1',
                # str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1',
                str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1',
                str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1',
                str(net_arch['pi']).replace(", ", "-") + '_1step_var0_1'],
    'Model': [mod + 'R_V00_PPO', mod + 'R_V1_PPO',
              # mod + 'R_V20_PPO', mod + 'R_V3_PPO',
              mod + 'R_V4_PPO', mod + 'R_V5_PPO']
}
sim_count = len(Training['Sim'])
'''Training = {
    'Sim': [pfad + '\RL_Sim_V0_inter.spp', pfad + '\RL_Sim_V1_inter.spp',
            pfad + '\RL_Sim_V2_inter.spp', pfad + '\RL_Sim_V3_inter.spp',
            pfad + '\RL_Sim_V5_sparse.spp'],
    'Logs': [erg + 'R_V0_PPO', erg + 'R_V1_PPO', erg + 'R_V2_PPO', erg + 'R_V3_PPO', erg + 'R_V4_PPO'],
    'Logname': ['512-256-128-128-64_1step_var0_1', '512-256-128-128-64_1step_var0_1', '512-256-128-128-64_1step_var0_1',
                '512-256-128-128-64_1step_var0_1', '512-256-128-128-64_1step_var0_1'],
    'Model': [mod + 'R_V0_PPO', mod + 'R_V1_PPO', mod + 'R_V2_PPO', mod + 'R_V3_PPO', mod + 'R_V4_PPO']
}'''
'''
Training = {
    'Sim': [pfad + '\RL_Sim_20230317_inter.spp', pfad + '\RL_Sim_20230317_sparse.spp',
            pfad + '\RL_Sim_20230317_inter.spp',
            pfad + '\RL_Sim_20230317_sparse.spp'],
    'Logs': [erg + 'R_In_PPO', erg + 'R_Sp_PPO', erg + 'R_In_PPO_LSTM', erg + 'R_Sp_PPO_LSTM'],
    'Logname': ['512-256-128-128-64_1step_var0_1', '512-256-128-128-64_1step_var0_1', '512-256-128-128-64_1step_var0_1',
                '512-256-128-128-64_1step_var0_1'],
    'Model': [mod + 'R_In_PPO', mod + 'R_Sp_PPO', mod + 'R_In_PPO_LSTM', mod + 'R_Sp_PPO_LSTM']
}'''
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
policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=net_arch)
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64, 32, 16], vf=[256, 128, 64, 32, 16]))
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[128, 64], vf=[128, 64]))
eval_freq = 3100
n_eval_episodes = 2
learning_rate = 5e-4
n_epochs = 10
n_steps = 384  # 1024 384
clip_range = 0.15
clip_range_vf = None
total_timesteps = 75000
visible = True  # False True
info_keywords = tuple(['Typ1', 'Typ2', 'Typ3', 'Typ4', 'Typ5', 'Warteschlangen', 'Auslastung'])
data = {'policy_kwargs': policy_kwargs, 'eval_freq': eval_freq, 'n_eval_episodes': n_eval_episodes,
        'learning_rate': learning_rate, 'n_epochs': n_epochs, 'n_steps': n_steps, 'clip_range': clip_range,
        'clip_range_vf': clip_range_vf, 'total_timesteps': total_timesteps, 'Sim': Training['Sim']}


# learning_rate-value
def lrsched():
    def reallr(progress):
        lr = learning_rate
        if progress < 0.5:
            lr = learning_rate * 0.8
        if progress < 0.3:
            lr = learning_rate * 0.6
        if progress < 0.1:
            lr = learning_rate * 0.4
        return lr

    return reallr


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


# clip-value
def clipsched():
    def realclip(progress):
        clip = clip_range
        if progress < 0.85:
            clip = clip_range
        if progress < 0.66:
            clip = clip_range  # * 0.8
        if progress < 0.33:
            clip = clip_range  # * 0.6
        return clip

    return realclip


'''for i in range(sim_count):  # sim_count  # 
    os.makedirs(Training['Logs'][i], exist_ok=True)
    with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Settings.txt', "w") as datei:
        # Die Werte in die Datei schreiben, einen pro Zeile
        for name, wert in data.items():
            datei.write(str(name) + ' = ' + str(wert) + "\n")
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=False)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i], info_keywords=info_keywords)
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i], log_path=Training['Logs'][i],
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_on_new_best=stop_callback)

    #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=Training['Logs'][i],
    #            learning_rate=linear_schedule(learning_rate),
    #            n_epochs=n_epochs, clip_range=linear_schedule(clip_range), gamma=0.995,
    #            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
    #            clip_range_vf=clip_range_vf,
    #            n_steps=n_steps, policy_kwargs=policy_kwargs)#

    #model.learn(total_timesteps=5000, callback=[rollout_callback, eval_callback],
    #            tb_log_name=Training['Logname'][i], progress_bar=True)
    #model.save(Training['Model'][i] + '\\train_model')
    #del model

    # Evaluiere Agent, Ergebnisse Dokumentieren
    #model = PPO.load(Training['Model'][i] + '\\best_model', env)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    # print(mean_reward)

    # Eval von 5 Plänen die nicht teil vom Training sind & Eval Rand actions
    open(Training['Logs'][i] + '\\' + Training['Logname'][i] + 'Testing_Testing.txt', "w")
    for j in range(5):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        while not done:
            steps += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            reward_sum += rewards
        with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + 'Testing_Testing.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
    # Eval von Random Actions
    open(Training['Logs'][i] + '\\' + Training['Logname'][i] + 'Testing_Random.txt', "w")
    for j in range(5):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        while not done:
            steps += 1
            action = random.randint(0, 4)
            obs, rewards, done, info = env.step(action)
            reward_sum += rewards
        with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + 'Testing_Random.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
    env.close()'''

for i in range(3,4):  # sim_count range(sim_count)
    os.makedirs(Training['Logs'][i] + '_LSTM', exist_ok=True)
    with open(Training['Logs'][i] + '_LSTM' + '\\' + Training['Logname'][i] + '_Settings.txt', "w") as datei:
        # Die Werte in die Datei schreiben, einen pro Zeile
        for name, wert in data.items():
            datei.write(str(name) + ' = ' + str(wert) + "\n")
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=False)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i] + '_LSTM', info_keywords=info_keywords)
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i] + '_LSTM',
                                 log_path=Training['Logs'][i] + '_LSTM',
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_on_new_best=stop_callback)

    '''model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=Training['Logs'][i] + '_LSTM',
                         learning_rate=linear_schedule(learning_rate), n_epochs=n_epochs,
                         clip_range=linear_schedule(clip_range),
                         device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
                         clip_range_vf=clip_range_vf,
                         n_steps=n_steps, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=[rollout_callback, eval_callback],
                tb_log_name=Training['Logname'][i], progress_bar=True)
    model.save(Training['Model'][i] + '_LSTM' + '\\train_model')
    del model'''

    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = RecurrentPPO.load(Training['Model'][i] + '_LSTM' + '\\best_model', env)
    '''
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    # print(mean_reward)
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

    # Eval von 5 Plänen die nicht teil vom Training sind & Eval Rand actions
    open(Training['Logs'][i] + '_LSTM' + '\\' + Training['Logname'][i] + 'Testing_Testing.txt', "w")
    # for j in range(sim_count):
    for j in range(5):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)
        while not done:
            steps += 1
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts,
                                                deterministic=True)
            obs, rewards, done, info = env.step(action)
            episode_starts = done
            reward_sum += rewards
        with open(Training['Logs'][i] + '_LSTM' + '\\' + Training['Logname'][i] + 'Testing_Testing.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
    # Eval von Random Actions
    open(Training['Logs'][i] + '_LSTM' + '\\' + Training['Logname'][i] + 'Testing_Random.txt', "w")
    for j in range(5):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        while not done:
            steps += 1
            action = random.randint(0, 4)
            obs, rewards, done, info = env.step(action)
            reward_sum += rewards
        with open(Training['Logs'][i] + '_LSTM' + '\\' + Training['Logname'][i] + 'Testing_Random.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
