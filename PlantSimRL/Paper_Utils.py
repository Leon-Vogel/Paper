import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorboard as tb
from matplotlib.ticker import MultipleLocator
from Paper_Utils_multiple import ergebnisse_return

fsize = 10
tsize = 10
tdir = 'in'
major = 5
minor = 3
style = 'default'  # 'default' helvetica


def ergebnis(x, y, names=None, title='title', yachse='Return', xachse='Episode', leg_pos='upper right'):
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

    # fig, ax = plt.subplots(figsize=(xsize, ysize))

    plt.figure(figsize=(xsize, ysize))
    plt.title(str(title), {'fontsize': plt.rcParams['font.size']})
    plt.grid(linestyle=':')

    running_avg = [[], [], [], []]
    for i in range(len(y)):
        N = len(y[i])
        # running_avg = np.empty(N)
        for t in range(N):
            running_avg[i].append(np.mean(y[i][max(0, t - 30):(t + 1)]))
            # plt.scatter(x, y, s=3, label='PPO_LSTM')
    x_1 = np.asarray(x)
    y_1 = np.asarray(running_avg)
    plt.plot(x_1[0], y_1[0], label=names[0], color='tab:blue')
    plt.plot(x_1[1], y_1[1], label=names[1], color='tab:orange')
    # R1 = pd.Series(data=running_avg[0], index=x[0])
    # R2 = pd.Series(data=running_avg[1], index=x[1])
    # plt.plot(R1, label=names[0])
    # plt.plot(R2, label=names[1])
    # plt.plot(x[1], running_avg[1], label=names[1])
    # plt.plot(x[2], running_avg[2], label=names[2])
    # plt.plot(x[3], running_avg[3], label=names[3])  # , 5, marker='x'

    # plt.annotate('Plotting style = ' + style, xy=(1, 0.05), ha='left', va='center')
    # plt.annotate('Figure size = ' + str(xsize) + ' x ' + str(ysize) + ' (in inches)', xy=(1, 0.045), ha='left',
    #             va='center')
    # plt.annotate('Font size = ' + str(fsize) + ' (in pts)', xy=(1, 0.04), ha='left', va='center')
    # plt.annotate('Tick direction = ' + tdir, xy=(1, 0.035), ha='left', va='center')
    # plt.annotate('Tick major, minor size = ' + str(major) + ' and ' + str(minor), xy=(1, 0.03), ha='left', va='center')

    ax = plt.gca()

    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    # ax.yaxis.set_minor_locator(MultipleLocator(10))
    # ax.set_xticklabels()
    ax.legend(loc=leg_pos, prop={
        'family': 'Helvetica'})
    plt.xlabel(xachse, labelpad=10)
    plt.ylabel(yachse, labelpad=10)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots\single_' + title.strip() + '.png', dpi=300, pad_inches=.1, bbox_inches='tight')


def ergebnisse(x1, y1, x2, y2, names=['test'], title='title', yachse='Return', xachse='Episode', leg_pos='upper right',
               auto_ylim=False):
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

    color = 'tab:red'
    fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    plt.title(str(title), {'fontsize': plt.rcParams['font.size']})
    ax1.set_xlabel(xachse)
    ax1.set_ylabel(yachse)

    ax1.grid(linestyle=':')

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
    x_2 = np.asarray(x2)
    y_2 = np.asarray(running_avg2)

    ax1.plot(x_1[0], y_1[0], label=names[0], color='tab:blue')
    ax1.plot(x_1[1], y_1[1], label=names[1], color='tab:orange')
    if len(x_1) > 2:
        ax1.plot(x_1[2], y_1[2], label=names[2], color='tab:green')
        ax1.plot(x_1[3], y_1[3], label=names[3], color='tab:red')
    # ax1.plot(x_1.T, y_1.T, label=names)
    ax1.tick_params(axis='y')
    ax1.set_xlabel(xachse, labelpad=10)
    ax1.set_ylabel(yachse, labelpad=10)
    if not auto_ylim:
        ax1.set_ylim(ymin=0, ymax=7500)
        # ax1.set_xlim(xmin=0, xmax=76000)
        ax1.yaxis.set_major_locator(MultipleLocator(1000))

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Plan Erfüllung')  # we already handled the x-label with ax1

    ax2.plot(x_2[0], y_2[0], label=names[0], color='tab:blue', linestyle='--', alpha=0.7)
    ax2.plot(x_2[1], y_2[1], label=names[1], color='tab:orange', linestyle='--', alpha=0.7)
    if len(x_2) > 2:
        ax2.plot(x_2[2], y_2[2], label=names[2], color='tab:green', linestyle='--', alpha=0.7)
        ax2.plot(x_2[3], y_2[3], label=names[3], color='tab:red', linestyle='--', alpha=0.7)
    # ax2.plot(x_2.T, y_2.T, label=names, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y')
    if not auto_ylim:
        ax2.set_ylim(ymin=0.7, ymax=1)
    ax1.legend(loc=leg_pos, prop={
        'family': 'Helvetica'}, ncol=2)
    # ax2.yaxis.tick_right()

    # plt.xlabel(xachse, labelpad=10)
    fig.tight_layout()
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

names = ['$R_1$', '$R_2$', '$R_3$', '$R_4$']
# names = ['R1', 'R2', 'R3', 'R4']
name = 'R00'
title = 'Return PPO'
yachse = 'Return'
'''x_rew = list(df['rollout/ep_rew_mean'].index.values)
y_rew = df['rollout/ep_rew_mean'].to_list()

ergebnis(x_rew, y_rew, name=name, title=title + '_single')'''
'''
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

    x_rew.append(list(df['rollout/ep_rew_mean'].index.values))
    y_rew.append(df['rollout/ep_rew_mean'].to_list())

ergebnisse(x_rew, y_rew, names=names, title=title, yachse=yachse)
'''

title = 'PPO'
yachse = 'Durchlaufzeit [s]'
x_0 = []
y_0 = []
y_1 = []
x_return_ppo = []
y_return_ppo = []
x_schl_ppo = []
y_schl_ppo = []
df_total = pd.DataFrame()
for i in range(len(names)):
    d = []  # {}
    for event in summary_iterator(pfad + R[0 + i] + N + F[0 + i]):
        d = []
        for value in event.summary.value:
            if value.HasField('simple_value') and event.step is not None:
                # d.append({'Run': names[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})
                d.append({'Run': names[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})

        # print(value.simple_value)
        if len(d) > 0:
            df = pd.DataFrame(d)
            df_total = pd.concat([df_total, df], ignore_index=False, sort=False)

# df_total.to_excel(title + "_Dlz.xlsx")
df_pivot = pd.pivot_table(df_total, values='Value', index=['Run', 'Step'], columns=['Tag'])
# df_pivot.to_excel(title + "_Dlz_pivot.xlsx")

df_pivot_tmp = df_pivot[df_pivot['Dlz/Typ1'].notnull()]
df_pivot_ret = df_pivot[df_pivot['rollout/ep_rew_mean'].notnull()]
df_pivot_sch = df_pivot[df_pivot['Mittelwert/Warteschlangen'].notnull()]
for i in range(len(names)):
    y_0.append(df_pivot_tmp.loc[(names[i])][['Dlz/Typ1', 'Dlz/Typ2', 'Dlz/Typ3', 'Dlz/Typ4',
                                             'Dlz/Typ5']].mean(axis=1).to_list())
    y_1.append(df_pivot_tmp.loc[(names[i])]['Plan/Anteil_fertigeProdukte'].to_list())
    x_0.append(df_pivot_tmp.loc[(names[i])]['Dlz/Typ1'].index.values.tolist())
    y_return_ppo.append(df_pivot_ret.loc[(names[i])]['rollout/ep_rew_mean'].to_list())
    x_return_ppo.append(df_pivot_ret.loc[(names[i])]['rollout/ep_rew_mean'].index.values.tolist())
    y_schl_ppo.append(df_pivot_sch.loc[(names[i])]['Mittelwert/Warteschlangen'].to_list())
    x_schl_ppo.append(df_pivot_sch.loc[(names[i])]['Mittelwert/Warteschlangen'].index.values.tolist())

ergebnisse(x_0, y_0, x_0, y_1, names=names, title=title, yachse=yachse, xachse='Step', leg_pos='upper left')

title = 'PPO LSTM'
yachse = 'Durchlaufzeit [s]'
x_0 = []
y_0 = []
y_1 = []
x_return_lstm = []
y_return_lstm = []
x_schl_lstm = []
y_schl_lstm = []
df_total = pd.DataFrame()
for i in range(len(names)):
    d = []  # {}
    for event in summary_iterator(pfad + R[4 + i] + N_LSTM[i] + F[4 + i]):
        d = []
        for value in event.summary.value:
            if value.HasField('simple_value') and event.step is not None:
                # d.append({'Run': names[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})
                d.append({'Run': names[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})

        # print(value.simple_value)
        if len(d) > 0:
            df = pd.DataFrame(d)
            df_total = pd.concat([df_total, df], ignore_index=False, sort=False)

# df_total.to_excel(title + "_Dlz.xlsx")
df_pivot = pd.pivot_table(df_total, values='Value', index=['Run', 'Step'], columns=['Tag'])
# df_pivot.to_excel(title + "_Dlz_pivot.xlsx")

df_pivot_tmp = df_pivot[df_pivot['Dlz/Typ1'].notnull()]
df_pivot_ret = df_pivot[df_pivot['rollout/ep_rew_mean'].notnull()]
df_pivot_sch = df_pivot[df_pivot['Mittelwert/Warteschlangen'].notnull()]
for i in range(len(names)):
    y_0.append(df_pivot_tmp.loc[(names[i])][['Dlz/Typ1', 'Dlz/Typ2', 'Dlz/Typ3', 'Dlz/Typ4',
                                             'Dlz/Typ5']].mean(axis=1).to_list())
    y_1.append(df_pivot_tmp.loc[(names[i])]['Plan/Anteil_fertigeProdukte'].to_list())
    x_0.append(df_pivot_tmp.loc[(names[i])]['Dlz/Typ1'].index.values.tolist())
    y_return_lstm.append(df_pivot_ret.loc[(names[i])]['rollout/ep_rew_mean'].to_list())
    x_return_lstm.append(df_pivot_ret.loc[(names[i])]['rollout/ep_rew_mean'].index.values.tolist())
    y_schl_lstm.append(df_pivot_sch.loc[(names[i])]['Mittelwert/Warteschlangen'].to_list())
    x_schl_lstm.append(df_pivot_sch.loc[(names[i])]['Mittelwert/Warteschlangen'].index.values.tolist())

ergebnisse(x_0, y_0, x_0, y_1, names=names, title=title, yachse=yachse, xachse='Step', leg_pos='upper left')

title = 'Return'
yachse = 'Return'
ergebnisse_return(x_return_ppo, y_return_ppo, x_return_lstm, y_return_lstm, names=names, title=title,
                  yachse=yachse, leg_pos='lower right')

title = 'Warteschlangen'
yachse = 'Produkte pro Förderstrecke'
ergebnisse_return(x_schl_ppo, y_schl_ppo, x_schl_lstm, y_schl_lstm, names=names, title=title,
                  yachse=yachse, leg_pos='upper right')

pfad = 'ergebnisse_Test1'
R = ['\R_V2_PPO', '\R_V1_PPO']
F = ['\events.out.tfevents.1678959333.DESKTOP-6FHK9F7.17876.0',
     '\events.out.tfevents.1678928371.DESKTOP-6FHK9F7.6944.1']
N = '\\512-256-128-128-64_1step_var0_1_2'

name_spa = ['$R_0$', '$R_2$']
# name_spa = ['R0', 'R2']
title = 'Dense und Sparse Reward'
yachse = 'Return'
x_0 = []
y_0 = []
x_1 = []
y_1 = []
df_total = pd.DataFrame()
for i in range(len(name_spa)):
    d = []  # {}
    for event in summary_iterator(pfad + R[0 + i] + N + F[0 + i]):
        d = []
        for value in event.summary.value:
            if value.HasField('simple_value') and event.step is not None:
                # d.append({'Run': names[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})
                d.append({'Run': name_spa[i], 'Step': event.step, 'Tag': value.tag, 'Value': value.simple_value})

        # print(value.simple_value)
        if len(d) > 0:
            df = pd.DataFrame(d)
            df_total = pd.concat([df_total, df], ignore_index=False, sort=False)

# df_total.to_excel(title + "_Dlz.xlsx")
df_pivot = pd.pivot_table(df_total, values='Value', index=['Run', 'Step'], columns=['Tag'])
# df_pivot.to_excel(title + "_Dlz_pivot.xlsx")

df_pivot_tmp = df_pivot[df_pivot['Plan/Anteil_fertigeProdukte'].notnull()]
df_pivot_ret = df_pivot[df_pivot['rollout/ep_rew_mean'].notnull()]
for i in range(len(name_spa)):
    y_0.append(df_pivot_ret.loc[(name_spa[i])]['rollout/ep_rew_mean'].to_list())
    x_0.append(df_pivot_ret.loc[(name_spa[i])]['rollout/ep_rew_mean'].index.values.tolist())
    y_1.append(df_pivot_tmp.loc[(name_spa[i])]['Plan/Anteil_fertigeProdukte'].to_list())
    x_1.append(df_pivot_tmp.loc[(name_spa[i])]['Plan/Anteil_fertigeProdukte'].index.values.tolist())

ergebnisse(x_0, y_0, x_1, y_1, names=name_spa, title=title, yachse=yachse, xachse='Step', leg_pos='upper left',
           auto_ylim=True)
