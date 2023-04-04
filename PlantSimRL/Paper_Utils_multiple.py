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


def ergebnisse_return(x1, y1, x2, y2, names=None, title='title', yachse='Return', xachse='Step', leg_pos='upper right'):
    # plt.style.use(style)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(xsize, ysize))
    #ax1.set_title('PPO')
    #ax2.set_title('PPO LSTM')

    # plt.figure(figsize=(xsize, ysize))
    ax1.set_title('PPO', {'fontsize': plt.rcParams['font.size']})
    ax1.grid(linestyle=':')
    ax2.set_title('PPO LSTM', {'fontsize': plt.rcParams['font.size']})
    ax2.grid(linestyle=':')

    running_avg1 = [[], [], [], []]
    for i in range(len(y1)):
        N = len(y1[i])
        # running_avg = np.empty(N)
        for t in range(N):
            running_avg1[i].append(np.mean(y1[i][max(0, t - 30):(t + 1)]))
            # plt.scatter(x, y, s=3, label='PPO_LSTM')
    running_avg2 = [[], [], [], []]
    for i in range(len(y2)):
        N = len(y2[i])
        # running_avg = np.empty(N)
        for t in range(N):
            running_avg2[i].append(np.mean(y2[i][max(0, t - 30):(t + 1)]))
            # plt.scatter(x, y, s=3, label='PPO_LSTM')
    x_1 = np.asarray(x1)
    y_1 = np.asarray(running_avg1)
    ax1.plot(x_1[0], y_1[0], label=names[0], color='tab:blue')
    ax1.plot(x_1[1], y_1[1], label=names[1], color='tab:orange')
    ax1.plot(x_1[2], y_1[2], label=names[2], color='tab:green')
    ax1.plot(x_1[3], y_1[3], label=names[3], color='tab:red')
    ax1.set_xlabel(xachse, labelpad=10)
    ax1.set_ylabel(yachse, labelpad=10)
    x_2 = np.asarray(x2)
    y_2 = np.asarray(running_avg2)
    ax2.axes.sharey(ax1)
    ax2.plot(x_2[0], y_2[0], label=names[0], color='tab:blue')
    ax2.plot(x_2[1], y_2[1], label=names[1], color='tab:orange')
    ax2.plot(x_2[2], y_2[2], label=names[2], color='tab:green')
    ax2.plot(x_2[3], y_2[3], label=names[3], color='tab:red')

    ax2.sharex(ax1)
    ax2.set_xlabel(xachse, labelpad=10)
    # ax = plt.gca()

    ax2.legend(loc=leg_pos, prop={
        'family': 'Helvetica'})
    #plt.xlabel(xachse, labelpad=10)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots\\' + title.strip() + '.png', dpi=300, pad_inches=.1, bbox_inches='tight')



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
N_LSTM = ['\[128-128-64]_1step_var0_1_1', '\[128-128-64]_1step_var0_1_1', '\[128-128-64]_1step_var0_1_2',
          '\[128-128-64]_1step_var0_1_1']
'''
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
# print(d)
df = pd.DataFrame.from_dict(d, orient='index')
df = df.transpose()'''
# print(df.keys())

names = ['R1', 'R2', 'R3', 'R4']
name = 'R00'
title = 'Return'
yachse = 'Return'

x_rew1 = []
y_rew1 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[0+i] + N + F[0 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew1.append(list(df['rollout/ep_rew_mean'].index.values))
    y_rew1.append(df['rollout/ep_rew_mean'].to_list())

x_rew2 = []
y_rew2 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew2.append(list(df['rollout/ep_rew_mean'].index.values))
    y_rew2.append(df['rollout/ep_rew_mean'].to_list())

ergebnisse_return(x_rew1, y_rew1, x_rew2, y_rew2, names=names, title=title, yachse=yachse, leg_pos='lower right')








title = 'Warteschlangen'
yachse = 'Produkte pro Förderstrecke'

x_rew1 = []
y_rew1 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[0+i] + N + F[0 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew1.append(list(df['Mittelwert/Warteschlangen'].index.values))
    y_rew1.append(df['Mittelwert/Warteschlangen'].to_list())

x_rew2 = []
y_rew2 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew2.append(list(df['Mittelwert/Warteschlangen'].index.values))
    y_rew2.append(df['Mittelwert/Warteschlangen'].to_list())

ergebnisse_return(x_rew1, y_rew1, x_rew2, y_rew2, names=names, title=title, yachse=yachse, leg_pos='upper right')









title = 'Durchlaufzeit'
yachse = 'Mittelwert Durchlaufzeit pro Produkt [s]'

x1_0 = []
y1_1 = []
y1_2 = []
y1_3 = []
y1_4 = []
y1_5 = []
y1_0 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[0+i] + N + F[0 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x1_0.append(df['Dlz/Typ1'].index.values)
    y1_1 = (df['Dlz/Typ1'].to_list())
    y1_2 = (df['Dlz/Typ2'].to_list())
    y1_3 = (df['Dlz/Typ3'].to_list())
    y1_4 = (df['Dlz/Typ4'].to_list())
    y1_5 = (df['Dlz/Typ5'].to_list())
    #  tmp = [(xv1 + xv2 + xv3 + xv4 + vx5) / 5 for xv1, xv2, xv3, xv4, vx5 in zip(*[x_1, x_2, x_3, x_4, x_5])]
    tmp = np.array([y1_1, y1_2, y1_3, y1_4, y1_5])
    y1_0.append(np.average(tmp, axis=0))

x2_0 = []
y2_1 = []
y2_2 = []
y2_3 = []
y2_4 = []
y2_5 = []
y2_0 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x2_0.append(df['Dlz/Typ1'].index.values)
    y2_1 = (df['Dlz/Typ1'].to_list())
    y2_2 = (df['Dlz/Typ2'].to_list())
    y2_3 = (df['Dlz/Typ3'].to_list())
    y2_4 = (df['Dlz/Typ4'].to_list())
    y2_5 = (df['Dlz/Typ5'].to_list())
    #  tmp = [(xv1 + xv2 + xv3 + xv4 + vx5) / 5 for xv1, xv2, xv3, xv4, vx5 in zip(*[x_1, x_2, x_3, x_4, x_5])]
    tmp = np.array([y2_1, y2_2, y2_3, y2_4, y2_5])
    y2_0.append(np.average(tmp, axis=0))

ergebnisse_return(x1_0, y1_0, x2_0, y2_0, names=names, title=title, yachse=yachse, leg_pos='upper right')






'''

title = 'Durchlaufzeit PPO'
yachse = 'Mittelwert Durchlaufzeit pro Produkt [s]'

x_0 = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []
y_5 = []
y_0 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[0+i] + N + F[0 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_0.append(df['Dlz/Typ1'].index.values)
    y_1 = (df['Dlz/Typ1'].to_list())
    y_2 = (df['Dlz/Typ2'].to_list())
    y_3 = (df['Dlz/Typ3'].to_list())
    y_4 = (df['Dlz/Typ4'].to_list())
    y_5 = (df['Dlz/Typ5'].to_list())
    #  tmp = [(xv1 + xv2 + xv3 + xv4 + vx5) / 5 for xv1, xv2, xv3, xv4, vx5 in zip(*[x_1, x_2, x_3, x_4, x_5])]
    tmp = np.array([y_1, y_2, y_3, y_4, y_5])
    y_0.append(np.average(tmp, axis=0))

ergebnisse(x_0, y_0, names=names, title=title, yachse=yachse)





title = 'Warteschlangen PPO'
yachse = 'Produkte pro Förderstrecke'

x_rew = []
y_rew = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[0+i] + N + F[0 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew.append(list(df['Mittelwert/Warteschlangen'].index.values))
    y_rew.append(df['Mittelwert/Warteschlangen'].to_list())

ergebnisse(x_rew, y_rew, names=names, title=title, yachse=yachse)
'''
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
'''
title = 'Return PPO LSTM'
yachse = 'Return'

x_rew = []
y_rew = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew.append(list(df['rollout/ep_rew_mean'].index.values))
    y_rew.append(df['rollout/ep_rew_mean'].to_list())

ergebnisse(x_rew, y_rew, names=names, title=title)






title = 'Durchlaufzeit PPO LSTM'
yachse = 'Mittelwert Durchlaufzeit pro Produkt [s]'

x_0 = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []
y_5 = []
y_0 = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_0.append(df['Dlz/Typ1'].index.values)
    y_1 = (df['Dlz/Typ1'].to_list())
    y_2 = (df['Dlz/Typ2'].to_list())
    y_3 = (df['Dlz/Typ3'].to_list())
    y_4 = (df['Dlz/Typ4'].to_list())
    y_5 = (df['Dlz/Typ5'].to_list())
    #  tmp = [(xv1 + xv2 + xv3 + xv4 + vx5) / 5 for xv1, xv2, xv3, xv4, vx5 in zip(*[x_1, x_2, x_3, x_4, x_5])]
    tmp = np.array([y_1, y_2, y_3, y_4, y_5])
    y_0.append(np.average(tmp, axis=0))

ergebnisse(x_0, y_0, names=names, title=title, yachse=yachse)




title = 'Warteschlangen PPO LSTM'
yachse = 'Produkte pro Förderstrecke'

x_rew = []
y_rew = []
for i in range(len(names)):
    d = {}
    for event in summary_iterator(pfad + R[4+i] + N_LSTM[i] + F[4 + i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew.append(list(df['Mittelwert/Warteschlangen'].index.values))
    y_rew.append(df['Mittelwert/Warteschlangen'].to_list())

ergebnisse(x_rew, y_rew, names=names, title=title, yachse=yachse)'''