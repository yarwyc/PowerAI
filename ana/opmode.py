# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:21:09 2020

@author: sdy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.manifold import TSNE

path = "D:/python/db/2019_09_12/tsne"
outline = pd.read_csv(path + "/outline_mm.csv", index_col=[0])
dt_idx = pd.to_datetime(outline.index, format='%Y_%m_%dT%H_%M_%S', exact=False)
outline.index = dt_idx
outline = outline[outline.index.month < 12]

bound_idx = list(outline.idxmin(axis=0)) + list(outline.idxmax(axis=0))
bound_idx = list(set(bound_idx))
n_bound = len(bound_idx)

avg_idx = []
for col in outline.columns:
    idx = outline.sort_values(col).index
    avg_idx.append(idx[len(idx) // 2])
n_avg = len(avg_idx)
minute_interval = 60
hour_interval = 2
oc_valid = (outline.index.minute % minute_interval == 0) * \
    (outline.index.hour % hour_interval == 0)
oc_idx = list(outline.index[oc_valid])
n_oc = len(oc_idx)

idx = bound_idx + avg_idx + oc_idx
modes = outline.loc[idx, :]
modes = modes * 1.4 - 0.2
idx = pd.DataFrame(data=list(range(len(idx))), index=idx, columns=['idx'])
idx['type'] = 2
idx.loc[idx.index[:n_bound], 'type'] = 0
idx.loc[idx.index[n_bound:n_bound + n_avg], 'type'] = 1

cct = pd.read_table(path + "/cct.txt", encoding='gbk', index_col=[0])
cct = cct.T
cct.drop(columns=cct.columns[[1,5,8]], inplace=True)
dt_idx = pd.to_datetime(cct.index, format='%Y_%m_%dT%H_%M_%S', exact=False)
cct.index = dt_idx
# modes = pd.concat([modes, cct], axis=1, join_axes=[modes.index])
modes = cct.loc[idx.index,:]

tsne = TSNE(random_state=6)
op_tsne = tsne.fit_transform(modes)


def scatter_key(ax, op_tsne, idx):
    op_i = idx.loc[idx['type'] == 0, 'idx']
    ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1], s=30,
               c='k', marker='^', label='extreme modes')
    op_i = idx.loc[idx['type'] == 1, 'idx']
    ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1], s=30,
               c='k', marker='o', label='typical modes')


def scatter_month(ax, op_tsne, idx):
    colors = ['m', 'b', 'g']
    for i in range(9, 12):
        op_i = idx.loc[(idx.index.month == i) & (idx.type == 2), 'idx']
        if(len(op_i) == 0):
            continue
        dt_i = idx.index[(idx.index.month == i) & (idx.type == 2)][0]\
            .strftime("%B")
        ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1],
                   c=colors[i - 9], label=dt_i)
    # scatter_key(ax, op_tsne, idx)
    ax.legend(loc='best')

def scatter_day(ax, op_tsne, idx):
    colors = plt.cm.rainbow(np.linspace(0, 1, 31))
    for i in range(1,32):
        op_i = idx.loc[(idx.index.day == i) & (idx.type == 2), 'idx']
        if(len(op_i) == 0):
            continue
        ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1],
                   c=colors[i-1], label='%d'%i)
    # scatter_key(ax, op_tsne, idx)
    ax.legend(loc='best', ncol=3)

def scatter_hour(ax, op_tsne, idx):
    global hour_interval
    colors = plt.cm.rainbow(np.linspace(0, 1, 24))
    op_i = idx.loc[idx.type == 2, 'idx']
    # ax.plot(op_tsne[op_i, 0], op_tsne[op_i, 1])
    # '''
    for i in range(0, 24, hour_interval):
        op_i = idx.loc[(idx.index.hour == i) & (idx.type == 2), 'idx']
        ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1],# s=10,
                   c=colors[i], label='%d:00' % i, alpha=0.7)
    # '''
    # scatter_key(ax, op_tsne, idx)
    ax.legend(loc='best', ncol=2)


def plot_dt(ax, op_tsne, idx, dt1, dt2):
    global hour_interval
    valid = (dt1 <= idx.index) & (idx.index < dt2)
    op_i = idx.loc[(idx.type == 2) & valid, 'idx']
    if(len(op_i) < 2):
        return
    # ax.plot(op_tsne[op_i, 0], op_tsne[op_i, 1], color='k')
    for i in range(len(op_i)-1):
        j, k = op_i[i], op_i[i+1]
        '''
        ax.arrow(op_tsne[j, 0], op_tsne[j, 1],
                 op_tsne[k, 0]-op_tsne[j, 0], op_tsne[k, 1]-op_tsne[j, 1],
                 length_includes_head = True, width=0.5,
                 fc='black', ec='black')
        '''
        ax.annotate("",
                xy=(op_tsne[k, 0], op_tsne[k, 1]),
                xytext=(op_tsne[j, 0], op_tsne[j, 1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="k"))
    colors = plt.cm.rainbow(np.linspace(0, 1, 24))
    for i in range(0, 24, hour_interval):
        op_i = idx.loc[(idx.index.hour == i) & (idx.type == 2) & valid, 'idx']
        ax.scatter(op_tsne[op_i, 0], op_tsne[op_i, 1], s=30, c=colors[i],\
                   marker='D')


fig = plt.figure(num=1, figsize=(8, 6))
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks([])
ax.set_yticks([])
scatter_month(ax, op_tsne, idx)
# scatter_day(ax, op_tsne, idx)
# scatter_hour(ax, op_tsne, idx)
# plot_dt(ax, op_tsne, idx,
#         datetime.datetime(2019, 9, 15), datetime.datetime(2019, 9, 16, 6))

modes_delta = modes.values[n_bound + n_avg + 1:, :] \
    - modes.values[n_bound + n_avg:-1, :]
modes_delta = np.linalg.norm(modes_delta, axis=1)
modes_delta /= np.max(modes_delta)
tsne_delta = op_tsne[n_bound + n_avg + 1:, :] - op_tsne[n_bound + n_avg:-1, :]
tsne_delta = np.linalg.norm(tsne_delta, axis=1)
tsne_delta /= np.max(tsne_delta)
