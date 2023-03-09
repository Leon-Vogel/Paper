from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment
import numpy as np
import os
import torch as T
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import functional as F
from stable_baselines3.common.env_checker import check_env
from CustomCallbacks import CustomCallback

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pfad = 'E:\\Studium\Projekt\Paper\PlantSimRL\simulations'
# pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\PPO_sb3'
logs = speicherort + '.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230301.spp'

Modell_laden = True  # True False

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=simulation,
                        socket=None, visible=False)
    env = Environment(plantsim)
    # print('Check env:')
    # check_env(env)
    # print('######### Check finished.')
    env = Monitor(env, logs)

    policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))

    if Modell_laden:
        model = PPO.load(speicherort + '\\best_model', env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=3e-4, n_epochs=10, clip_range=0.15,
                    device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=0.15,
                    n_steps=384, policy_kwargs=policy_kwargs)

    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=speicherort, log_path=logs, eval_freq=2000,
                                 n_eval_episodes=2, callback_on_new_best=stop_callback)

    # 50000 timestep dauern ~ 1std.
    model.learn(total_timesteps=70000, callback=[rollout_callback, eval_callback],
                tb_log_name="first_run", progress_bar=True)
    model.save(speicherort + '\ppo_model')

    env.close()

# Parameter für alle Trainings festlegen, Netz, lernrate etc.
# Directory für Ergebnisse definieren (evtl mit Dictionary?)

# Für alle intermediate Rewardfunktionen: Lade Sim

# Erstelle PPO Agent in eigener Directory, trainiere für 100.000 Schritte, speichern etc.
# Evaluiere Agent, Ergebnisse Dokumentieren

# Erstelle PPO LSTM Agent in eigener Directory, trainiere für 100.000 Schritte, speichern etc.
# Evaluiere Agent, Ergebnisse Dokumentieren


# Für alle sparse Rewardfunktionen: Lade Sim

# Erstelle PPO Agent in eigener Directory, trainiere für 100.000 Schritte, speichern etc.
# Evaluiere Agent, Ergebnisse Dokumentieren

# Erstelle PPO LSTM Agent in eigener Directory, trainiere für 100.000 Schritte, speichern etc.
# Evaluiere Agent, Ergebnisse Dokumentieren
