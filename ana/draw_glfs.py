# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font10 = FontProperties(fname=r"C:/Windows/Fonts/STKAITI.TTF", size=10)
path = r'D:/python/db/2018_11/sst/'
num = 17
names = ['黑龙江.鸡西二热厂/15kV.#1机',\
			'黑龙江.鸡西二热厂/15kV.#2机']
names = []
if len(names) == 0:
	idx = pd.read_table(path+"/round_%04d.glf"%num, encoding='gbk', header=None, sep=' ', index_col=1)
	names = list(idx.index)

df = pd.DataFrame(data = None, index = range(num+1), columns = names, dtype = np.float32)
for i in range(1, num+1):
	cur = pd.read_table('%s/round_%04d.glf' % (path, i),\
					encoding='gbk', header=None, sep=' ', index_col=1)
	for name in df.columns:
		if name in cur.index:
			df.loc[i, name] = cur.loc[name, 2]
			df.loc[i-1, name] = cur.loc[name, 2] - cur.loc[name, 3]
for i in range(num-1, -1, -1):
	for j in range(len(df.columns)):
		if np.isnan(df.iloc[i, j]):
			df.iloc[i, j] = df.iloc[i+1, j]

df = df.append(df.loc[num] - df.loc[0], ignore_index=True)
increase_columns = df.loc[num+1] > 0
df1 = df.loc[:, increase_columns]
df1 = df1.sort_values(by=[num+1], axis=1, ascending=False)
legends = []
for i in range(len(df1.columns)):
    legends.append(df1.columns[i] + '(' + str(round(df1.loc[num+1][i], 2)) + ')')
plt.figure(figsize=(8, 6))
plt.plot(df1.iloc[:num+1, :])
plt.legend(legends, prop=font10, bbox_to_anchor=(1.05, 1.02))
plt.subplots_adjust(right=0.6)
plt.savefig(path+"/line+.png")
plt.show()

df2 = df.loc[:, -increase_columns]
df2 = df2.sort_values(by=[num+1], axis=1, ascending=True)
legends = []
for i in range(len(df2.columns)):
    legends.append(df2.columns[i] + '(' + str(round(df2.loc[num+1][i], 2)) + ')')
plt.figure(figsize=(8, 6))
plt.plot(df2.iloc[:num+1, :])
plt.legend(legends, prop=font10, bbox_to_anchor=(1.05, 1.02))
plt.subplots_adjust(right=0.6)
plt.savefig(path+"/line-.png")
plt.show()
