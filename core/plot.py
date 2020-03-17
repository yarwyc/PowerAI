# coding: gbk

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:/Windows/Fonts/STSONG.TTF", size=12)

from power.elem import load_elem_info,load_st_info
from power.ed import load_yzr

def plot_embedding_2d(X, names, branches=None, title=None):
	x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
	X = (X - x_min) / (x_max - x_min)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	for i in range(X.shape[0]):
		ax.text(X[i, 0], X[i, 1], names[i], fontdict={'size': 10},fontproperties=font)
	if(branches is not None):
		br = branches[branches[:,3]>0.]
		ibus = br[:,0].astype(np.int)
		jbus = br[:,1].astype(np.int)
		rx = np.sqrt(np.sum(br[:,2:4]**2, axis=1))
		cc = (np.max(rx)-rx)/(np.max(rx)-np.min(rx))
		cc=cc**20*0.8+0.1
		for i in range(br.shape[0]):
			plt.plot([X[ibus[i]-1][0],X[jbus[i]-1][0]],[X[ibus[i]-1][1],X[jbus[i]-1][1]],color='%.2f'%cc[i],fontproperties=font)
	if(title is not None):
		plt.title(title,fontproperties=font)
	plt.show()

def plot_X(X, names):
	x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
	x = (X - x_min) / (x_max - x_min)
	fig = plt.figure()
	fig.scatter(x[:, 0], x[:, 1])
	# ax = fig.add_subplot(1, 1, 1)
	# for i in range(X.shape[0]):
		# fig.scatter(X[i, 0], X[i, 1])
		# ax.text(X[i, 0], X[i, 1], names[i], fontdict={'size': 10}, fontproperties=font)
	
def plot_stations(stations, label=False):
	# x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
	# X = (X - x_min) / (x_max - x_min)
	colors = ['r', 'c', 'g']
	sizes = [60, 50, 25]
	markers = ['x', 'o', 'o']
	for g in range(3)[::-1]:
		d = stations[stations.group==g]
		plt.scatter(d['x'], d['y'], c=colors[g], s=sizes[g],
			  marker=markers[g])
	if label:
		d = stations[(stations.group==0)|(stations.group==1)]
		for i in d.index:
			plt.text(d['x'][i], d['y'][i], i.split('.')[1],
				   fontdict={'size': 8}, fontproperties=font)
	'''
	for _,d in stations.iterrows():
		g = int(d['group'])
		plt.scatter(d['x'], d['y'], c=colors[g], s=sizes[g],
			  marker=markers[g])
	'''

def plot_lines(lines):
	for _,d in lines.iterrows():
		if d['vl']<500:
			continue
		plt.plot([d['x1'],d['x2']],[d['y1'],d['y2']],'r-')

def get_index(all_names, sub_names):
	idx = []
	for sub in sub_names:
		if(sub in all_names):
			idx.append(np.where(all_names==sub)[0][0])
	return idx

def gen_stations_pos(path, area):
	st = load_st_info(path+"/st_info.dat")
	st = st[st.area==area]
	y,z,z_r,st_names = load_yzr('%s/st_y.dat'%path, '%s/st_map.dat'%path, gen_num_thres=-1)
	st_names = st_names[:,0]
	idx = get_index(st_names, st.index)
	names = st_names[idx]
	sub_zr = z_r[idx,:][:,idx]
	'''
	tsne = TSNE(n_components=2, metric='precomputed', method='barnes_hut',
			perplexity = 20.0, early_exaggeration=12.0,
			n_iter=500, n_iter_without_progress=30,
			learning_rate=0.1, random_state=np.random.randint(65536))
	'''
	tsne = TSNE(n_components=2, metric='precomputed', method='exact',
				n_iter=2000, n_iter_without_progress=100,
				learning_rate=0.1, random_state=np.random.randint(65536))
	X = tsne.fit_transform(sub_zr)
	x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
	X = (X - x_min) / (x_max - x_min)

	g1 = [3]		#zhongzhou
	g2 = get_index(names,st[(st["vl"]==1000)|(st["vl"]==500)].index.values)
	g3 = get_index(names,st[st["vl"]==220].index.values)
	data = np.zeros((names.shape[0],3))
	data[:,:2] = X
	data[g2,2] = 1
	data[g3,2] = 2
	data[g1,2] = 0
	stations = pd.DataFrame(data, index=names, columns=['x','y','group'])
	return stations,sub_zr,names

def gen_lines_pos(path, stations, area):
	elems = load_elem_info(path+"/elem_info.dat")
	elems = elems[elems.type==2]
	elems = elems[elems.area==area]
	xy = np.empty((elems.shape[0],5)) + np.NaN
	for i,name in enumerate(elems.index):
		xy[i,4] = elems.vl[name]
		st1,st2 = elems.station[name].split('_')
		if st1 not in stations.index or st2 not in stations.index:
			continue
		xy[i,0] = stations.x[st1]
		xy[i,1] = stations.y[st1]
		xy[i,2] = stations.x[st2]
		xy[i,3] = stations.y[st2]
	lines = pd.DataFrame(xy, index=elems.index,
					columns=['x1','y1','x2','y2','vl'])
	return lines

def plot_contour(X,Y,f):
	x = np.linspace(0,1,101)
	y = np.linspace(0,1,101)
	z = np.zeros((101,101))
	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			d = np.sqrt((X-xx)**2+(Y-yy)**2)
			valid = (d<0.1)
			d = d[valid]
			sub = f[valid]
			if d.shape[0] == 0:
				continue;
			z[j][i] = np.sum(sub)/d.shape[0]
	
	C = plt.contour(x, y, z, 20, alpha = 0.75, cmap = plt.cm.hot)
	plt.clabel(C, inline = True, fontsize = 8)

if __name__ == '__main__':
	path = 'D:/PSA_src/power_tools/data/EMT_10'

	# stations = pd.read_csv(path+"/hn_st.csv", encoding="gbk", index_col=0)
	stations,sub_zr,names = gen_stations_pos(path, 157)
	lines = gen_lines_pos(path, stations, 157)
	
	plot_stations(stations)
	X = []
	Y = []
	f = []
	# plot_lines(lines)
	for i,res in res2.iterrows():
		if i not in lines.index:
			continue
		X.append(lines.x1[i])
		Y.append(lines.y1[i])
		# X.append((lines.x1[i]+lines.x2[i])/2)
		# Y.append((lines.y1[i]+lines.y2[i])/2)
		f.append(res2.duration[i])
	for i,res in res98.iterrows():
		if i not in lines.index:
			continue
		X.append(lines.x2[i])
		Y.append(lines.y2[i])
		# X.append((lines.x1[i]+lines.x2[i])/2)
		# Y.append((lines.y1[i]+lines.y2[i])/2)
		f.append(res98.duration[i])
	X = np.array(X)
	Y = np.array(Y)
	f = np.array(f)
	plot_contour(X,Y,f)
