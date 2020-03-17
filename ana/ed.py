# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import os
from itertools import combinations


def n1_df(path):
    """
    Collect n-1 ed results for 1 LF, and put them into a DataFrame.
    Args:
        path: str. Contain ed results, and also output path.
    Return:
        df: DataFrame. All results.
    """
    df = pd.DataFrame()
    files = os.listdir(path)
    for i, file_name in enumerate(files):
        if(i % 10 == 0):
            print("%d / %d" % (i, len(files)))
        if(file_name.startswith('n1_') == False):
            continue
        sub = pd.read_table(os.path.join(path, file_name),
                            sep=' ', encoding='gbk', index_col=[0])
        df = pd.concat([df, sub], axis=1)
    st_names = pd.read_table(os.path.join(path, 'idx_name.txt'),
                             sep=' ', encoding='gbk', index_col=[0])
    new_idx = []
    for idx in df.index:
        i, j = int(idx.split('_')[0]), int(idx.split('_')[1])
        new_idx.append(
            '_'.join([st_names['name'][i], st_names['name'][j]]))
    df.index = new_idx
    ss = np.sum(df > 10) > 0
    df = df.drop(labels=ss[ss].index, axis=1)
    # df.to_csv(os.path.join(path, 'all.csv'), float_format='%.8f')
    np.savez(os.path.join(path, 'all.npz'), index=df.index,
             columns=df.columns, datas=df.values)
    return df

def stat_prop(df):
    """
    Get df's statistics property
    Args:
        df: DataFrame. Index for st_st, column for n-1 line name
    Return:
        df_prop: DataFrame.
            Index for st_st;
            Column in ['min', 'max', 'max_min', 'mu', 'std']
    """
    df_prop = pd.concat([df.min(axis=1), df.max(axis=1)], axis=1)
    df_prop.columns = ['min', 'max']
    df_prop['max_min'] = df_prop['max'] - df_prop['min']
    df_prop['mu'] = df.mean(axis=1)
    df_prop['std'] = df.std(axis=1)
    return df_prop

def minset_greedy(df, thr=0.05):
    """
    Get df's minimum set that covers all AC line N-1 using greedy method
    Args:
        df: DataFrame. Index for st_st, column for n-1 line name
    Return:
        index: station names of minset
    """
    ret = []
    delta = df.values
    delta = np.abs(delta - delta[:,0:1]) / (delta + 1e-8)
    delta = (delta[:, 1:] > thr)
    covered = np.zeros((delta.shape[1],), dtype = np.bool_)
    while(True):
        remain_sum = np.sum(delta[:, ~covered], axis = 1)
        i = remain_sum.argmax()
        if(remain_sum[i] == 0):
            break
        # print(df.index[i], np.sum(delta[i,~covered]), np.sum(covered))
        ret.append(i)
        covered += delta[i,:]
    return df.index[ret]

def minset_brute(df, thr=0.05, max_amount=10):
    """
    Get df's minimum set that covers all AC line N-1 using brute force
    Args:
        df: DataFrame. Index for st_st, column for n-1 line name
    Return:
        index: station names of minset
    """
    delta = df.values
    delta = np.abs(delta - delta[:,0:1]) / (delta + 1e-8)
    delta = (delta[:, 1:] > thr)
    for m in range(1, max_amount+1):
        print("m =", m)
        for ret in combinations(range(delta.shape[0]), m):
            ret = list(ret)
            if(np.all(np.sum(delta[ret, :], axis=0)) == True):
                print("brute done!")
                return df.index[ret]
    print("brute failed.")
    return None

if __name__ == '__main__':
    path = "db/2019_09_12/2019_12_15T11_00_00/n1"
    # df = n1_df(path)
    arch = np.load(os.path.join(path, 'all.npz'), allow_pickle=True)
    df = pd.DataFrame(data=arch['datas'],
                      index=arch['index'],
                      columns=arch['columns'])
    # df_prop = stat_prop(df)
    # minset = minset_greedy(df)
    minset = minset_brute(df, max_amount = 2)
    # print(minset)
