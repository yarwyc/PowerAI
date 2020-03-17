# coding: gbk

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame


def output_one(path):
    index = [os.path.dirname(path).split("\\")[-1]]

    idx = pd.read_table(path, encoding='gbk')
    up = idx[idx.iloc[:, 0].str.contains('//')].index[0]
    down = idx[idx.iloc[:, 0].str.contains('</CCTOUT::>')].index[0]

    data_one_all = []
    for j in range(up + 1, down):
        data_line = idx.iloc[j, :][0]
        data = [[data_line.split(" ")[3], data_line.split(" ")[4]]]
        data_one_all.append(DataFrame(data=data[0][1], index=[data[0][0]], columns=[index]))
    data_merge_single = pd.concat(data_one_all, axis=0)
    return data_merge_single


def select_cct(in_path, out_path, fault_list):
    filelist = []
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file[-3:] == 'res':
                filelist.append(os.path.join(root, file))

    data_all = []
    for i in range(len(filelist)):
        one = output_one(filelist[i])
        data_all.append(one)
    data_merge_all = pd.concat(data_all, axis=1)

    true_fault_list = []
    for fl in fault_list:
        if fl in data_merge_all.index:
            true_fault_list.append(fl)
        else:
            print(fl + '不在最后级联后的DataFrame中')
    if not true_fault_list:
        print('所有故障都不在最后级联后的DataFrame中，生成级联后的DataFrame')
        data_merge_all.to_csv(out_path + '\\res.txt', sep="\t", encoding="gbk")
    else:
        truedata = data_merge_all.loc[true_fault_list, :]
        truedata.to_csv(out_path + '\\res.txt', sep="\t", encoding="gbk")


def select_sst(in_path, out_path):
    filelist = []
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file[-3:] == 'res':
                filelist.append(os.path.join(root, file))

    data1_all = []

    # data2_all = []
    for i in range(len(filelist)):

        index = [os.path.dirname(filelist[i]).split("\\")[-1]]

        idx = pd.read_table(filelist[i], encoding='gbk')
        df = idx[idx.iloc[:, 0].str.contains('辽宁_黑龙江')]
        # data2_line = idx[idx.iloc[:, 0].str.contains('辽宁增长模式')].iloc[0, 0]

        if df.shape == (1, 1):
            data1 = [[df.iloc[0, 0].split(" ")[5], df.iloc[0, 0].split(" ")[6]]]
            data1_all.append(DataFrame(data=data1, index=index, columns=['辽宁_黑龙江_al', '辽宁_黑龙江_f']))

        else:
            data1_all.append(DataFrame(data=[(np.nan, np.nan)], index=index, columns=['辽宁_黑龙江_al', '辽宁_黑龙江_f']))

        # if data2_line != '':
        #     data2 = [[data2_line.split(" ")[6], data2_line.split(" ")[11]]]
        #     data2_all.append(DataFrame(data=data2, index=index, columns=['辽宁增长模式_vsl', '辽宁增长模式_cur_power']))
    data_merge1 = pd.concat(data1_all, axis=0)
    # data_merge2 = pd.concat(data2_all, axis=0)
    data_merge1.to_csv(out_path + '\\辽宁_黑龙江.txt', sep="\t")
    # data_merge2.to_csv(out_path + '\\辽宁增长模式.txt', sep="\t")


def select_curve(in_path, out_path):
    filelist = []
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file[-3:] == 'res':
                filelist.append(os.path.join(root, file))
    filelist.sort()

    data_all = []

    for i in range(len(filelist)):
        index = [os.path.dirname(filelist[i]).split("\\")[-1]]
        print(index)
        idx = pd.read_table(filelist[i], encoding='gbk', header=None)
        up_df = idx[idx.iloc[:, 0].str.contains('<CURVEOUT:: soft=>')]
        down_df = idx[idx.iloc[:, 0].str.contains('</CURVEOUT::>')]
        up_shape = up_df.shape
        down_shape = down_df.shape
        if up_shape != (0, 1) and down_shape != (0, 1):
            up = up_df.index[0]
            down = down_df.index[0]
            newdf = idx.iloc[up + 1:down, :]
            text = newdf.iloc[2, 0]
            data = [[text.split(' ')[-2], text.split(' ')[-1]]]
            data_all.append(DataFrame(data=data, index=index, columns=['第一列', '第二列']))
        else:
            data_all.append(DataFrame(data=[(np.nan, np.nan)], index=index, columns=['第一列', '第二列']))

    data_merge = pd.concat(data_all, axis=0)
    data_merge.to_csv(out_path + '\\curve.txt', sep="\t")


# path1 = r"F:\2019_9-11\res"
# path2 = r'F:\2019_9-11\res'
# select_cct(path1, path2)
#
# path11 = r'F:\gd_data2\sst'
# path22 = r'F:\gd_data2'
# select_sst(path11, path22)

# path111 = r'F:\test'
# path222 = r'F:\test'
# select_curve(path111, path222)
select_cct(r'F:\2019_9-11\test', r'F:\2019_9-11\test', ['东北.董辽一线', '东北.燕董一线', '哈哈'])
