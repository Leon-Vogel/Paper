import os
import sys
import time

import numpy as np
import torch as T
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.logger import configure

from plantsim.plantsim import Plantsim
from ps_environment_sb3 import Environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pfad = 'D:\\Studium\Projekt\Paper\PlantSimRL\simulations'
speicherort = 'tmp\PPO_sb3_test'
logs = speicherort + '.\logs\\'
os.makedirs(logs, exist_ok=True)
simulation = pfad + '\RL_Sim_20230301.spp'

train_episodes = 100
Modell_laden = False  # True False

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=simulation,
                        socket=None, visible=False)
    env = Environment(plantsim)
    # print('Check env:')
    # check_env(env)
    # print('######### Check finished.')
    env = Monitor(env, logs)

    policy_kwargs = dict(activation_fn=T.nn.ReLU, net_arch=[256, 128, 64])

    if Modell_laden:
        model = PPO.load(speicherort + 'ppo_model', env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs, learning_rate=5e-4, n_epochs=5, clip_range=0.15,
                    device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
                    n_steps=512, policy_kwargs=policy_kwargs)

    total_steps = 0
    last_episode_starts = False
    new_logger = configure(logs, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    for i in range(train_episodes):
        model.start_time = time.time_ns()
        n_steps = 0
        score = 0
        best = -2000
        observation = env.reset()
        observation = observation.reshape((1, 36))
        done = False
        model.rollout_buffer.reset()
        model.policy.set_training_mode(False)
        while not done:
            with T.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(observation, model.device)
                actions, values, log_probs = model.policy(obs_tensor)
            actions = actions.cpu().numpy()
            new_obs, rewards, done, infos = env.step(actions)
            n_steps += 1
            total_steps += 1
            score += rewards
            model.rollout_buffer.add(observation, actions, rewards, last_episode_starts, values, log_probs)
            observation = new_obs.reshape((1, 36))
            last_episode_starts = done

            if done:
                if score > best:
                    model.save(speicherort + '\ppo_model')
                new_obs = new_obs.reshape((1, 36))
                with T.no_grad():
                    # Compute value for the last timestep
                    values = model.policy.predict_values(obs_as_tensor(new_obs, model.device))
                model.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
                # model._update_current_progress_remaining(total_steps, total_timesteps)

                # Display training infos
                time_elapsed = max((time.time_ns() - model.start_time) / 1e9, sys.float_info.epsilon)
                fps = int(n_steps / time_elapsed)
                model.logger.record("time/iterations", i, exclude="tensorboard")
                model.logger.record("rollout/ep_rew_mean", score)
                model.logger.record("rollout/ep_len_mean", n_steps)
                model.logger.record("time/fps", fps)
                model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                model.logger.record("time/total_timesteps", total_steps, exclude="tensorboard")
                model.logger.dump(step=n_steps)

                model.rollout_buffer.full = True
                model.train()

    # model.save(speicherort + '\ppo_model')
    env.close()
