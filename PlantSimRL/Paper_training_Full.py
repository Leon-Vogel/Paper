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
pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
erg = 'ergebnisse\\'
mod = 'models\\'
Training = {
    'Sim': [pfad + '\RL_Sim_20230310.spp', pfad + '\R2.spp', pfad + '\R3.spp', pfad + '\RL_Sim_20230310.spp',
            pfad + '\R2.spp', pfad + '\R3.spp'],
    'Logs': [erg + 'PPO_R1', erg + 'PPO_R2', erg + 'PPO_R3', erg + 'PPO_LSTM_R1', erg + 'PPO_LSTM_R2',
             erg + 'PPO_LSTM_R3'],
    'Model': [mod + 'PPO_R1', mod + 'PPO_R2', mod + 'PPO_R3', mod + 'PPO_LSTM_R1', mod + 'PPO_LSTM_R2',
              mod + 'PPO_LSTM_R3']
}
# pfad = 'E:\\Studium\Projekt\Paper\PlantSimRL\simulations'
pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\PPO_sb3'
logs = speicherort + '.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230310.spp'
policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))

for i in range(3):
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=False)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i])
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i], log_path=logs, eval_freq=2000,
                                 n_eval_episodes=3, callback_on_new_best=stop_callback)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=3e-4, n_epochs=15, clip_range=0.15,
                device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=0.15,
                n_steps=1024, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=70000, callback=[rollout_callback, eval_callback],
                tb_log_name="first_run", progress_bar=True)
    model.save(Training['Model'][i] + '\\train_model')
    del model

    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = PPO.load(Training['Model'][i])
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

# Erstelle PPO LSTM Agent in eigener Directory, trainiere für 100.000 Schritte, speichern etc.
for i in range(3, 6):
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=False)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i])
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i], log_path=logs, eval_freq=2000,
                                 n_eval_episodes=3, callback_on_new_best=stop_callback)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=3e-4, n_epochs=15,
                         clip_range=0.15,
                         device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=0.15,
                         n_steps=1024, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=70000, callback=[rollout_callback, eval_callback],
                tb_log_name="first_run", progress_bar=True)
    model.save(Training['Model'][i] + '\\train_model')
    del model

    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = PPO.load(Training['Model'][i])
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
