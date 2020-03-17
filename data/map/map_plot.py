# coding: gbk

import numpy as np
import pandas as pd
from power.elem import load_st_info


def write_gen_contour(values, path, res_path, type_name):
    st_info = load_st_info(path + "/ednet/st_info.dat")
    st_pos = pd.read_csv(path + "/ednet/st_pos.csv",
                         index_col=0, encoding='GBK')
    st_info = pd.concat([st_info, st_pos], axis=1)
    value_count = np.zeros((st_info.shape[0], 2))
    for name, value in values:
        for i, id in enumerate(st_info.index):
            if id in name:
                # row['value'] += value
                # row['count'] += 1
                value_count[i][0] += value
                value_count[i][1] += 1
                break

    f = open(res_path + '/' + type_name + '.dat', 'w')
    i = 0
    for idx, row in st_info.iterrows():
        if np.isnan(row['lng']) or value_count[i][1] == 0:
            i += 1
            continue
        f.write('\t{\"lng\":%f,\"lat\":%f,\"count\":%f},\n'
                % (row['lng'], row['lat'], value_count[i][0] / value_count[i][1]))
        i += 1
    f.close()