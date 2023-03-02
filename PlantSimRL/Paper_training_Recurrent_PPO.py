import os
import sys
import time
from copy import deepcopy

import torch as T
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment
from utils import plot_learning_curve
from CustomCallbacks import CustomCallback


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\Recurrent_PPO_sb3'
logs = speicherort + '.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230301.spp'

Modell_laden = False  # True False

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
        model = RecurrentPPO.load(speicherort + '\\best_model', env)
    else:
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=5e-4, n_epochs=10,
                             clip_range=0.15, device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
                             n_steps=384, policy_kwargs=policy_kwargs, clip_range_vf=0.15)

    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=speicherort, log_path=logs, eval_freq=2000,
                                 n_eval_episodes=2, callback_on_new_best=stop_callback)

    # 50000 timestep dauern ~ 1std.
    model.learn(total_timesteps=70000, callback=[eval_callback],  # rollout_callback,
                tb_log_name="first_run", progress_bar=True)
    model.save(speicherort + '\\recurrent_ppo_model')

    env.close()

