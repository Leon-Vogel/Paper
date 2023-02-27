
from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment
import numpy as np
import os
import torch as T
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\Recurrent_PPO_sb3'
logs = speicherort+'.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230227.spp'

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=simulation,
                        socket=None, visible=False)
    env = Environment(plantsim)
    env = Monitor(env, logs)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logs,
                         device=T.device('cuda:0'if T.cuda.is_available() else'cpu'),
                         n_steps=320)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=speicherort, log_path=logs, eval_freq=50,
                                 n_eval_episodes=3, callback_on_new_best=stop_callback)

    # um ein Training fortzuführen: reset_num_timesteps=False
    model.learn(total_timesteps=50, callback=[eval_callback, stop_callback], tb_log_name="first_run")

    '''x, y = ts2xy(load_results(logs), 'timesteps')
    logger.configure(logs)
    logger.record('timesteps', x[-1])
    logger.record('mean_reward', y[-1])
    logger.dumpkvs()'''
    # Visualisiere die Trainingsergebnisse mit Tensorboard
    os.system(f"tensorboard --logdir {logs} --port 6006")

    env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    model.save(speicherort+'ppo_recurrent')
    del model

    model = RecurrentPPO.load(speicherort+'ppo_recurrent')
    obs = env.reset()
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        #env.render()

