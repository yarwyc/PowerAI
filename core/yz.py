# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import warnings

from common.time_util import timer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

EPS = 1e-8


def generate_y_matrix(power, island, node_type='bus', dtype=np.float32,
                      on_only=True, ignore_ground_branch=True, x_only=True):
    if node_type == 'bus':
        if 'island' not in power.buses.columns:
            raise ValueError("Island info not generated.")
        nodes = power.data['bus'].index[power.data['bus'].island == island].values
        columns = ['mark', 'ibus', 'jbus', 'r', 'x']
        branches = np.vstack((power.data['acline'][columns].values,
                              power.data['transformer'][columns].values)).astype(dtype)
    elif node_type == 'station':
        if 'island' not in power.stations.columns:
            raise ValueError("Island info not generated.")
        nodes = power.stations.index[power.stations.island == island].values
        columns = ['mark', 'st_i', 'st_j', 'r', 'x']
        branches = power.data['acline'][columns].values.astype(dtype)
    else:
        return None
    if on_only:
        branches = branches[branches[:, 0] == 1.]
    if ignore_ground_branch:
        branches = branches[branches[:, 1] != branches[:, 2]]
    else:
        raise NotImplementedError("Ground branches are not considered.")
    valid = np.isin(branches[:, [1, 2]], nodes).all(axis=1)
    branches = branches[valid]

    if branches.shape[0] == 0:
        raise ValueError(
            "Branches is empty, while node size = %d" %
            len(nodes))

    branches = branches.copy()
    nodes = list(set(branches[:, [1, 2]].astype(np.int32).flatten()))
    n = len(nodes)
    node_idx = pd.Series(data=range(n), index=nodes)
    branches[:, 1] = node_idx.loc[branches[:, 1]].values
    branches[:, 2] = node_idx.loc[branches[:, 2]].values
    if x_only:
        gb = -1. / (branches[:, 4] + EPS)  # jb = -1/jx
        y = np.zeros((n, n), dtype=dtype)
    else:
        gb = np.vectorize(np.complex)(branches[:, 3].astype(dtype),
                                      branches[:, 4].astype(dtype))
        gb = 1. / (gb + EPS)  # g+jb=1/(r+jx)
        y = np.zeros((n, n), dtype=gb.dtype)
    for i, [ii, jj] in enumerate(branches[:, [1, 2]].astype(np.int32)):
        y[ii][jj] = y[jj][ii] = y[ii][jj] - gb[i]
        # y[i, j] = -(g+jb) in non-diagonal element
    for i in range(y.shape[0]):
        y[i, i] = 0.0
        y[i, i] = -np.sum(y[i, ]) * 1.01
    # y[0][0] = y[0][0] * 1.01
    return pd.DataFrame(data=y, index=nodes, columns=nodes)


def calc_z_from_y(y):
    if not isinstance(y, pd.DataFrame) and not isinstance(y, np.array):
        raise NotImplementedError("%s not supported." % type(y))
    if y.shape[0] > 1000:
        warnings.warn("It would be time-consuming: y.shape=%d." % y.shape[0])
    z = np.linalg.inv(y)
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(data=z, index=y.index, columns=y.columns)
    return z


def calc_ed_from_z(z, indices=None):
    if indices is None:
        nodes = z.index.to_list()
        zz = z.values
    else:
        indices = np.array(indices)
        nodes = list(set(indices.flatten()))
        node_idx = pd.Series(data=range(len(nodes)), index=nodes)
        zz = z.loc[nodes, nodes].values
    if len(nodes) == 0:
        raise ValueError("Node set is empty.")
    elif len(nodes) > 1000:
        warnings.warn("It would be time-consuming: node size= %d." % len(nodes))

    ed = np.zeros((zz.shape[0], zz.shape[0]))
    for i in range(0, zz.shape[0] - 1):
        j = np.arange(i + 1, zz.shape[0])
        ed[i, j] = np.vectorize(np.abs)(zz[i, i] + zz[j, j] - zz[i, j] - zz[j, i])
        # ed[j, i] = ed[i, j]
    ed = ed + ed.T
    if indices is None or indices.ndim == 1:
        return pd.DataFrame(data=ed, index=nodes, columns=nodes)
    elif indices.ndim == 2:
        assert indices.shape[1] == 2
        i = node_idx.loc[indices[:, 0]].values
        j = node_idx.loc[indices[:, 1]].values
        return ed[i, j]
    return None


def calc_ed_from_power(power, island, node_type='bus', dtype=np.float32,
                       on_only=True, ignore_ground_branch=True, x_only=True,
                       indices=None):
    y = generate_y_matrix(power, island, node_type=node_type, dtype=dtype,
                          on_only=on_only, ignore_ground_branch=ignore_ground_branch,
                          x_only=x_only)
    z = calc_z_from_y(y)
    return calc_ed_from_z(z, indices)


def ed_map_tsne(ed, n_dim=2, **kwargs):
    assert isinstance(ed, pd.DataFrame) and ed.shape[0] == ed.shape[1]
    tsne = TSNE(n_components=n_dim, metric='precomputed', **kwargs)
    x = tsne.fit_transform(ed.values)
    x -= np.median(x, axis=0)
    return x


def group_kmeans(ed, n):
    assert isinstance(ed, pd.DataFrame) and ed.shape[0] == ed.shape[1]
    kmeans = KMeans(n_clusters=n)
    clf = kmeans.fit(ed.values)
    groups = {}
    for i in range(n):
        groups[i] = ed.index[np.where(clf.labels_ == i)[0]].to_list()
    return groups


if __name__ == '__main__':
    from core.power import Power

    path = 'D:/PSA_src/psa/localdata/0913/data'
    fmt = 'on'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=False)

    island = 0
    with timer('Generate Y matrix'):
        y = generate_y_matrix(power, island, node_type='station', x_only=False)
    with timer('Calculate Z from Y'):
        z = calc_z_from_y(y)
    with timer('Calculate ED matrix'):
        indices = power.stations.index[power.stations.island == island][:100].to_numpy()
        ed = calc_ed_from_z(z, indices=indices)
    # with timer('ED map to 2D by t-SNE'):
    #     x = ed_map_tsne(ed)
    # with timer('Group by kmeans'):
    #     groups = group_kmeans(ed, 10)
