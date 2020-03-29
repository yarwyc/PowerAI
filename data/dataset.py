# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import pandas as pd
import numpy as np
import glob
from io import StringIO

from core.power import Power
from core.power_def import format_key, file_format, \
    model_column, restore_column, \
    index_dict, name_index_dict, get_all_column
from common.time_util import timer
from common.efile_util import read_efile


def pack_data(path, file_name, fmt='on', types=None, combine_model=True,
              model_columns=None, restore_columns=None, ori_order=True):
    format_keys = format_key()
    if not types:
        types = format_keys['types']
    if not restore_columns:
        restore_columns = restore_column()
    types = [t for t in types if t in restore_columns]
    all_columns = get_all_column(types, file_format()[fmt])
    name_indices = name_index_dict()
    if combine_model:
        if not model_columns:
            model_columns = model_column()
        models = {}
    for t in types:
        miss = [col for col in restore_columns[t] if col not in all_columns[t]]
        restore_columns[t] = [col for col in restore_columns[t]
                              if col in all_columns[t] and col not in name_indices[t]]
        restore_columns[t] = restore_columns[t] + ['flag']
        if combine_model:
            miss.extend([col for col in model_columns[t] if col not in all_columns[t]])
            model_columns[t] = [col for col in model_columns[t]
                                if col in all_columns[t] and col not in name_indices[t]]
        if miss:
            print('[%s] data miss: ' % t, miss)

    data = dict([(t, {}) for t in types])
    for d in os.listdir(path):
        if not os.path.exists(os.path.join(path, d, 'LF.L1')):
            continue
        power = Power(fmt=fmt)
        power.load_power(os.path.join(path, d), fmt=fmt, station=False)
        for t in types:
            power.data[t].set_index(name_indices[t], inplace=True)
            if ori_order and 'ori_order' in power.data[t]:
                data[t][d] = power.data[t][restore_columns[t] + ['ori_order']]
            else:
                data[t][d] = power.data[t][restore_columns[t]]
            if combine_model:
                if t in models:
                    idx = power.data[t].index.difference(models[t].index)
                    models[t] = models[t].append(power.data[t].loc[idx,
                                                                   model_columns[t]])
                else:
                    models[t] = power.data[t][model_columns[t]]

    package = {}
    for t in types:
        package[t] = pd.concat(data[t].values(), keys=data[t].keys())
        if combine_model and t in models:
            package['model_' + t] = models[t]
    hdf = pd.HDFStore(file_name, 'w', complevel=9, complib='blosc')
    for k in package:
        hdf.put(key=k, value=package[k])
    hdf.close()


def get_package_info(file_name):
    hdf = pd.HDFStore(file_name, 'r')
    has_model = False
    for k in hdf.keys():
        if k.startswith('/model_'):
            has_model = True
    dirs = [idx for idx in hdf['/bus'].index.unique(0)] if '/bus' in hdf.keys() else []
    hdf.close()
    return has_model, dirs


def unpack_data(path, file_name, fmt='on', types=None, models=None, dirs=None):
    if not os.path.exists(path):
        os.mkdir(path)
    hdf = pd.HDFStore(file_name, 'r')
    if not types:
        types = format_key()['types']
    types = [t for t in types if t in restore_column()]
    if not models:
        models = {}
        for k in hdf.keys():
            if k.startswith('/model_'):
                models[k[7:]] = hdf[k]
    if not dirs:
        dirs = [idx[0] for idx in hdf['/bus'].index] if '/bus' in hdf.keys() else []
    indices = index_dict()

    for d in dirs:
        power = Power(fmt)
        for t in types:
            power.data[t] = hdf.get(t).loc[d]
            if t in models:
                columns = models[t].columns.difference(power.data[t].columns)
                power.data[t] = power.data[t].join(models[t][columns])
            for i, n in enumerate(power.data[t].index.names):
                power.data[t][n] = power.data[t].index.get_level_values(i)
            if t in indices:
                power.data[t].set_index(indices[t], drop=False, inplace=True)
                power.data[t].sort_index(inplace=True)
        power.generate_mdc_version_outline()
        power.save_power(os.path.join(path, d), fmt=fmt)


def collect_learning_data(path, etype, columns=None):
    print('collecting data...')
    file_name = etype + '.dat'
    files = glob.glob(os.path.join(path, '**', file_name), recursive=True)
    total = len(files)
    if total == 0:
        print('No data file [%s]' % file_name)
        return None
    data = {}
    interval = max(int(total / 100), 1)
    for i, f in enumerate(files):
        if i % interval == 0:
            print('\r%d / %d (%.2f)' % (i, total, 100 * i / total), end='')
        df = pd.read_table(f, sep='\s+', encoding='gbk', index_col=[0],
                           usecols=None if columns is None else ['name']+columns)
        name = f.split(os.sep)[-2]
        data[name] = df.astype(np.float32)
    print('\nconcat, unstack and save...')
    data = pd.concat(data.values(), keys=data.keys())
    for col in data.columns:
        sub = data[col].unstack()
        np.savez(os.path.join(path, etype + '_' + col + '.npz'),
                 data=sub.values, times=sub.index, elems=sub.columns)
    return data


def collect_learning_res(path, rtypes):
    print('collecting result...')
    files = glob.glob(os.path.join(path, '**', '*.res'), recursive=True)
    total = len(files)
    if total == 0:
        print('No res file')
        return None
    data = dict([(r, {}) for r in rtypes])
    interval = max(int(total / 100), 1)
    for i, f in enumerate(files):
        if i % interval == 0:
            print('\r%d / %d (%.2f)' % (i, total, 100 * i / total), end='')
        dfs = read_efile(f, rtypes.keys(), rtypes)
        name = f.split(os.sep)[-2]
        for r in dfs:
            dfs[r].set_index(rtypes[r][0], inplace=True)
            data[r][name] = dfs[r].astype(np.float32)
    print('\nconcat, unstack and save...')
    res = {}
    for r in data:
        res[r] = pd.concat(data[r].values(), keys=data[r].keys())
        for col in res[r].columns:
            sub = res[r][col].unstack()
            sub.to_csv(os.path.join(path, col + '.txt'), sep="\t", encoding="gbk")
    return res


if __name__ == '__main__':
    path = 'D:/PSASP_Pro/online/package'
    file_name = 'pack.h5'
    fmt = 'on'
    """
    with timer('Pack power'):
        pack_data(path, os.path.join(path, file_name), fmt='on')
    with timer('Get infos of package'):
        model, data = get_package_info(os.path.join(path, file_name))
    print(model, data)
    dir = '2019_12_23T13_10_00'
    with timer('Unpack power'):
        unpack_data(os.path.join(path, 'out'),
                    os.path.join(path, file_name), fmt='on', dirs=[dir])
    """

    path = 'D:/python/db/2019_09_12/data/'
    iwant = {'generator': ['p', 'v'],
             'station': ['pl', 'ql'],
             'dcline': None}
    for etype in iwant:
        with timer('Collect data [%s] for [%s]' % (path, etype)):
            data = collect_learning_data(path, etype, iwant[etype])
    """

    path = 'D:/python/db/2019_09_12/data/'
    iwant = {'CCTOUT': ['name', 'cct', 'times']}
    collect_learning_res(path, iwant)
    """
