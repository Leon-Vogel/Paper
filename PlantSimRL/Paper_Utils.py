import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorboard as tb
from matplotlib.ticker import MultipleLocator

fsize = 10
tsize = 10
tdir = 'in'
major = 5
minor = 3
style = 'default'  # 'default' helvetica


def ergebnisse(x, y, name='test'):
    #plt.style.use(style)

    N = len(y)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(y[max(0, t - 30):(t + 1)])


    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor

    xsize = 6
    ysize = 3

    # fig, ax = plt.subplots(figsize=(xsize, ysize))

    plt.figure(figsize=(xsize, ysize))

    plt.scatter(x, y, s=7, label='PPO_LSTM')
    plt.title(str(name))
    plt.plot(x, running_avg, label='Running_Avg')  # , 5, marker='x'

    plt.annotate('Plotting style = ' + style, xy=(1, 0.05), ha='left', va='center')
    plt.annotate('Figure size = ' + str(xsize) + ' x ' + str(ysize) + ' (in inches)', xy=(1, 0.045), ha='left',
                 va='center')
    plt.annotate('Font size = ' + str(fsize) + ' (in pts)', xy=(1, 0.04), ha='left', va='center')
    plt.annotate('Tick direction = ' + tdir, xy=(1, 0.035), ha='left', va='center')
    plt.annotate('Tick major, minor size = ' + str(major) + ' and ' + str(minor), xy=(1, 0.03), ha='left', va='center')

    ax = plt.gca()

    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.legend()
    plt.xlabel('$Episode in Training$', labelpad=10)
    plt.ylabel('$Return$', labelpad=10)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots\professional_plot.png', dpi=300, pad_inches=.1, bbox_inches='tight')


pfad = 'ergebnisse_Test3'
R = ['\R_V00_PPO', '\R_V1_PPO', '\R_V4_PPO', '\R_V5_PPO', '\R_V00_PPO_LSTM', '\R_V1_PPO_LSTM', '\R_V4_PPO_LSTM',
     '\R_V5_PPO_LSTM']
F = ['\events.out.tfevents.1679086287.DESKTOP-6FHK9F7.19296.0',
     '\events.out.tfevents.1679095374.DESKTOP-6FHK9F7.19296.1',
     '\events.out.tfevents.1679103875.DESKTOP-6FHK9F7.19296.2',
     '\events.out.tfevents.1679111124.DESKTOP-6FHK9F7.19296.3',
     '\events.out.tfevents.1679185588.DESKTOP-6FHK9F7.18940.0',
     '\events.out.tfevents.1679194755.DESKTOP-6FHK9F7.18940.1',
     '\events.out.tfevents.1679202988.DESKTOP-6FHK9F7.18940.2',  # '\[128-128-64]_1step_var0_1_2'!!!!
     '\events.out.tfevents.1679210498.DESKTOP-6FHK9F7.18940.3']
N = '\[128-128-64]_1step_var0_1_1'
# ergebnisse_Test3\R_V00_PPO
# experiment_id = pfad + R[0] + N + '\events.out.tfevents.1679086287.DESKTOP-6FHK9F7.19296.0'
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
# print(df)

d = {}
for event in summary_iterator(pfad + R[0] + N + F[0]):
    for value in event.summary.value:
        # print(value.tag)
        if value.HasField('simple_value'):
            if value.tag in d.keys():
                d[value.tag].append(value.simple_value)
            else:
                d.update(
                    {
                        str(value.tag): [value.simple_value]
                    }
                )
                # print(value.simple_value)
print(d)

df = pd.DataFrame.from_dict(d, orient='index')
df = df.transpose()
print(df.keys())



ergebnisse(list(df['rollout/ep_rew_mean'].index.values), df['rollout/ep_rew_mean'].to_list(), name='PPO_R1')
'''Index(['Dlz/Typ1', 'Dlz/Typ2', 'Dlz/Typ3', 'Dlz/Typ4', 'Dlz/Typ5',
       'Mittelwert/Auslastung', 'Mittelwert/Warteschlangen',
       'Plan/Anteil_fertigeProdukte', 'Varianz/Auslastung',
       'Varianz/Warteschlangen', 'rollout/ep_len_mean', 'rollout/ep_rew_mean',
       'time/fps', 'train/approx_kl', 'train/clip_fraction',
       'train/clip_range', 'train/entropy_loss', 'train/explained_variance',
       'train/learning_rate', 'train/loss', 'train/policy_gradient_loss',
       'train/value_loss', 
       'eval/mean_ep_length', 'eval/mean_reward'],
      dtype='object')
      '''
# print(df)
# df.to_csv('Raw_Data.csv', index=True)
