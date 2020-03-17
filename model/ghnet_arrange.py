# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from core.power_yz import calc_ed_from_power
from core.power import load_station_info, Power


def get_index(names, subs):
    """
    Get subs indices in names.

    :param names: 1D np.array(str).
    :param subs: 1D np.array(str).
    :return: [indices].
    """
    ret = []
    for sub in subs:
        idx = np.where(names == sub)[0]
        if len(idx) > 0:
            ret.append(idx[0])
        # else:
        # 	print(sub)
    return ret


def group_kv220(areas, st_info, ed):
    """
    Group 220kV stations.
    Assign 220kV to the nearest 500kV by electric distance,
    from electric distance to layer dict.

    :param areas: [int]. Specify areas to group, -1 for all.
    :param st_info: st_info.
    :param ed: pd.DataFrame. Electric distance matrix.
    :return: {upper_idx: [lower_indices]}.
    """
    layer = {}
    for area in areas:
        sub_st = st_info[st_info["area"] == area] if area > 0 else st_info
        lower_names = sub_st[sub_st["vl"] <= 220].index.values
        lower_names = ed.index[ed.index.isin(lower_names)]
        upper_names = sub_st[st_info["vl"] >= 500].index.values
        upper_names = ed.columns[ed.columns.isin(upper_names)]
        sub_ed = ed.loc[lower_names, upper_names]
        for lower, upper in sub_ed.idxmin(axis=1).iteritems():
            if upper in layer.keys():
                layer[upper].append(lower)
            else:
                layer[upper] = [lower]
    return layer


def group_kv500_kmeans(areas, st_info, ed, file_name):
    """
    Group 500kV stations by KMeans, from electric distance to g500.txt.

    :param areas: [int]. Specify areas (provinces) to group.
    :param st_info: st_info.
    :param ed: pd.DataFrame. Electric distance matrix.
    :param file_name: str. g500.txt.
    """
    f = open(file_name, 'w')
    f.write('id names\n')
    for area in areas:
        valid = (st_info["area"] == area) & (st_info["vl"] >= 500)
        names = st_info[valid].index.values
        names = ed.index[ed.index.isin(names)]
        sub_ed = ed.loc[names, names]
        if sub_ed.shape[0] == 0:
            continue
        n_group = int(sub_ed.shape[0] / 10) + 1
        clf = KMeans(n_clusters=n_group)
        clf.fit(sub_ed.values)
        # print(clf.labels_)
        res = {}
        for i in range(n_group):
            res[i] = []
        for i, label in enumerate(clf.labels_):
            res[label].append(sub_ed.index[i])
        for i in range(n_group):
            f.write("%d_%d " % (area, i + 1) + "+".join(res[i]) + "\n")
    f.close()


def f_name_split(x): return x.split('+')


def group_kv500(file_name, ed):
    """
    Group 500kV stations, from g500.txt to layer dict.

    :param file_name: str.
    :param ed: pd.DataFrame. Electric distance matrix.
    :return: {upper_idx: [lower_indices]}.
    """
    layer = {}
    centers = []
    groups = pd.read_table(file_name, encoding='gbk', sep=' ',
                           index_col=0, converters={'names': f_name_split})
    for names in groups['names'].values:
        # idx = get_index(st_names, names)
        names = ed.index[ed.index.isin(names)]
        sub_ed = ed.loc[names, names]
        centers.append(sub_ed.sum().idxmin())
    return layer, centers


def group_province(st_info, centers, provinces):
    """
    Group 500kV station center to province.

    :param st_info: st_info.
    :param centers: [str]. Center station names.
    :param provinces: {no: name}.
    :return: {upper_idx: [lower_indices]}.
    """
    layer = {}
    for c in centers:
        area = provinces[st_info['area'][c]]
        if area in layer.keys():
            layer[area].append(c)
        else:
            layer[area] = [c]
    return layer


def trans_layer(layer, names):
    """
    Translate layer from indices to name.

    :param layer: {upper: [lower]}.
    :param names: 1D np.array(str).
    :return: {upper_name: [lower_names]}.
    """
    ret = {}
    for key, value in layer.items():
        # print(type(key))
        if isinstance(key, np.int64):
            ret[names[key]] = names[value]
        elif isinstance(key, str):
            ret[key] = value
    return ret


def write_layers(path, layers):
    """
    Write layers to file.

    :param path: str.
    :param layers: [{upper_name: [lower_names]}].
    """
    with open('%s/ghnet.dat' % path, 'w') as f:
        f.write('layer upper lower\n')
        for i, layer in enumerate(layers):
            for key, value in layer.items():
                f.write('%d %s %s\n' % ((i + 1), key, '+'.join(value)))


def write_ed(path, layers):
    """
    Write ed_info.dat.
    Deprecated!!!

    :param path: str.
    :param layers: [{upper_name: [lower_names]}].
    """
    with open('%s/ed_info.dat' % path, 'w') as f:
        f.write('layer i j\n')
        for i, layer in enumerate(layers):
            if i == 0 or i == 1:
                for key, value in layer.items():
                    for v in value:
                        if key != v:
                            f.write('%d %s %s\n' % ((i + 1), key, v))
            elif i == 2:
                names2 = []
                for key, value in layer.items():
                    names2.extend(value)
                for j in range(len(names2)):
                    for k in range(j):
                        f.write('%d %s %s\n' % ((i + 1), names2[j], names2[k]))


def write_edinfo_from_ghnet(path, centers, level=1):
    """
    Write ed_info.dat from ghnet.dat.
    Deprecated!!!
    """
    ghnet = pd.read_table("%s/ghnet.dat" % path, encoding='gbk', sep=' ')
    ghnet = ghnet[ghnet['layer'] >= level]

    f = open('%s/ed_info.dat' % path, 'w')
    f.write('layer i j\n')
    for _, sub in ghnet.iterrows():
        i = sub['layer']
        upper = sub['upper']
        lowers = sub['lower'].split('+')
        if i <= 2:
            for lower in lowers:
                if upper != lower:
                    f.write('%d %s %s\n' % (i, upper, lower))
        else:
            if upper in centers:
                upper = centers[upper]
            for lower in lowers:
                if lower in centers:
                    lower = centers[lower]
                if upper != lower:
                    f.write('%d %s %s\n' % (i, upper, lower))
    f.close()


if __name__ == '__main__':
    areas = [21, 22, 23, 24]
    # areas = [42, 43]
    path = "D:/python/db/2019_09_12/db_2019_11_15T10_00_00/"

    power = Power(fmt='on')
    power.load_power(path, fmt='on', lp=False, st=False, station=True)
    island = power.get_largest_island()
    ed = calc_ed_from_power(power, island, node_type='station',
                            on_only=False, x_only=False)
    index = power.stations.loc[ed.index, 'ori_name']
    ed.index = index
    ed.columns = index
    st_info = load_station_info("%s/st_info.dat" % path)
    st_info["area"].replace(12, 11, inplace=True)
    st_info["area"].replace(18, 11, inplace=True)

    provinces = {1: 'GD',
                 10: 'HB', 20: 'DB', 30: 'HD', 40: 'HZ', 50: 'XB', 80: 'XN',
                 11: 'JingJinJi', 12: 'Tianjin', 13: 'Hebei', 14: 'Shanxi', 15: 'Neimeng', 16: 'Shandong', 18: 'Jibei',
                 21: 'Liaoning', 22: 'Jilin', 23: 'Heilongjiang', 24: 'Mengdong',
                 41: 'Henan', 42: 'Hubei', 43: 'Hunan', 44: 'Chongqing', 45: 'Sichuan', 46: 'Jiangxi'}

    layer1 = group_kv220([-1], st_info, ed)
    # layer1 = trans_layer(layer1, st_names)

    group_kv500_kmeans(areas, st_info, ed, "%s/g500.txt" % path)
    layer2, centers = group_kv500("%s/g500.txt" % path, ed)
    # layer2 = trans_layer(layer2, st_names)

    layer3 = group_province(st_info, centers, provinces)

    layer4 = {}
    # layer4[provinces[10]] = [provinces[11],provinces[13],provinces[14],provinces[15],provinces[16]]
    # layer4[provinces[40]] = [provinces[41],
    #                          provinces[42], provinces[43], provinces[46]]
    # layer4[provinces[80]] = [provinces[44],provinces[45]]
    layer4[provinces[20]] = [provinces[21], provinces[22], provinces[23]]

    # layer5 = {}
    # layer5[provinces[1]] = [provinces[10],provinces[40],provinces[80]]

    write_layers(path, [layer1, layer2, layer3, layer4])
    # write_layers(path, [layer1,layer2,layer3,layer4])
