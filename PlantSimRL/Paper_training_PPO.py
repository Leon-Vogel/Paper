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

# pfad = 'E:\\Studium\Projekt\Paper\PlantSimRL\simulations'
pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\PPO_sb3'
logs = speicherort + '.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230310.spp'

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
        model = PPO.load(speicherort + '\\best_model', env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=3e-4, n_epochs=10, clip_range=0.15,
                    device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'), clip_range_vf=0.15,
                    n_steps=384, policy_kwargs=policy_kwargs)
    # mehrere episoden für generalisierung


    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=speicherort, log_path=logs, eval_freq=2000,
                                 n_eval_episodes=3, callback_on_new_best=stop_callback)

    # 50000 timestep dauern ~ 1std.
    model.learn(total_timesteps=70000, callback=[rollout_callback, eval_callback],
                tb_log_name="first_run", progress_bar=True)
    model.save(speicherort + '\ppo_model')

    env.close()











    # Visualisiere die Trainingsergebnisse mit Tensorboard
    # os.system(f"tensorboard --logdir {logs} --port 6006")

    '''env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    model.save(speicherort + 'ppo')
    del model

    model = PPO.load(speicherort + 'ppo')
    obs = env.reset()
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        # env.render()'''
