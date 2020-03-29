# coding: gbk

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from models.ghnet_model import GHNet
from models.ghnet_data import GHData
		

def fourier(X_, y_):
	valid = (np.isnan(y_)==False)
	X = X_[valid, :]
	y = y_[valid]
	# y -= np.mean(y)
	km = 200
	step = 0.005
	for d in range(1):
		yk = np.zeros((km,))
		for k in range(km):
			yk[k] = np.abs(np.mean(y*np.exp(0-2j*np.pi*X[:,d]*k*step)))
		# plt.plot(np.arange(km)*step, yk, color=(1.0-0.1*d, 0.0, 0.1*d))
		plt.scatter(np.arange(km)*step, yk)

if __name__ == '__main__':

	path = "/home/sdy/python/db/2018_11"
	if os.name=='nt':
		path = "d:/python/db/2018_11"
	input_dic = {'generator':['p','v']}
	"""
	input_dic = {'generator':['p','v'],
	  			'station':['pg', 'pl','ql'],
	  			'dcline':['p','q','acu'],
				'ed':['ed']}
	"""

	net = GHNet("inf", input_dic)
	net.load_net(path+"/db_2018_11_27T10_00_00")

	data_set = GHData(path,
					path+"/db_2018_11_27T10_00_00",
					net.input_layer)
	data_set.load_x()
	data_set.load_y('cct')
	data_set.normalize()
	# data_set.split_dataset_dt(dt_train_begin = datetime.datetime(2018,11,1),
	# 					   dt_train_end = datetime.datetime(2018,11,21)-datetime.timedelta(seconds=1),
	# 					   dt_test_begin = datetime.datetime(2018,11,21),
	# 					   dt_test_end = datetime.datetime(2018,11,22)-datetime.timedelta(seconds=1),
	# 					   validation_perc = 0.5)

	X = data_set.input_datas.values.copy()
	X = X - np.mean(X, axis = 0)
	pca = PCA(n_components = 0.9)
	X_new = pca.fit_transform(X)
	y = data_set.y.values[:,0]
	fourier(X_new, y)