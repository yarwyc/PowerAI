# -*- coding: utf-8 -*-
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
from itertools import product


def KL_uniform(features, n_cut=5):
    n_sample = features.shape[0]
    n_feature = features.shape[1]
    pi = 1 / n_cut**n_feature
    qi = 1 / n_sample
    # kl = n_sample*pi*np.log2(pi/qi)
    kl = n_sample * pi * np.log2(pi / qi)
    return kl


def dist_center(features):
    n_feature = features.shape[1]
    feature_center = np.mean(features, axis=0)
    feature_center = feature_center.reshape((1, n_feature))
    dist = np.sqrt(np.sum((features - feature_center)**2, axis=1))
    return dist


def dist_uniform():
    n_feature = 6
    n_cut = 10
    p = np.linspace(-1, 1, n_cut + 1)
    dist = []
    for i in product(p, repeat=n_feature):
        dist.append(np.sqrt(np.sum(np.array(i)**2)))
    return np.array(dist)


if __name__ == '__main__':
    # dist = dist_center(features_all)
    dist_u = dist_uniform()
