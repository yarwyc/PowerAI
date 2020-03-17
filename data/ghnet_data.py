# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import datetime

from model.ghnet_model import GHNet
from core.power import load_elem_info


class GHData(object):

    """
    Dataset class for GHNet

    Attributes:
        data_path: str. *.npz
        model_path: str. Contains ghnet.dat, elem_info.dat, st_info.dat, ed_info.dat...
        input_layer: [(type, name)].
        input_data: DataFrame. Model input data
        ori_data: DataFrame. Original data
        y: DataFrame. Labels(multiple columns)
        sample_prop: DataFrame. Properties of dt, real, role
        column_valid: np.array.
        column_min: np.array.
        column_max: np.array.
        data_sets: dict {id: (data, labels, indices)}.

    """

    def __init__(self, data_path, model_path, input_layer):
        self.data_path = data_path
        self.model_path = model_path
        self.input_layer = input_layer

        self.input_data = None
        self.ori_data = None
        self.y = None
        self.sample_prop = None
        self.column_valid = None
        self.column_min = None
        self.column_max = None

        self.data_sets = {}

    def load_x(self, x_ratio_thr=0.2):
        """
        Load input_data, sample_prop

        :param x_ratio_thr: float. Invalid column if valid ratio < x_ratio_thr
        """
        print("loading data...")
        n_column = len(self.input_layer)
        indices = -np.ones((n_column,), dtype=np.int32)
        types = set([t for (t, elem) in self.input_layer])
        datas = []
        begin = 0
        for t in types:
            arch = np.load('%s/%s.npz' % (self.data_path, t), allow_pickle=True)
            data = pd.DataFrame(data=arch['datas'].T,
                                index=arch['times'],
                                columns=arch['elems'],
                                dtype=np.float32)
            for i, elem in enumerate(data.columns):
                if (t, elem) in self.input_layer:
                    indices[self.input_layer.index((t, elem))] = begin + i
            begin += len(data.columns)
            datas.append(data)
            print(t, data.shape)
        self.input_data = pd.concat(datas, axis=1)
        self.input_data = self.input_data.iloc[:, indices]
        # still the original size, where invalid columns should be datas[-1]
        self.input_data.drop_duplicates(inplace=True)
        self.input_data.sort_index(inplace=True)
        self.input_data.columns = [
            '_'.join((t, elem)) for t, elem in self.input_layer]
        self.ori_data = self.input_data.copy()
        valid = np.sum(self.input_data.isna() == False, axis=0)
        valid = (valid / self.input_data.shape[0]) > x_ratio_thr
        self.column_valid = (indices >= 0) & valid.values
        dt = pd.to_datetime(self.input_data.index,
                            format='%Y_%m_%dT%H_%M_%S',
                            exact=False).values
        real = (self.input_data.index.str.len() == 19)
        role = np.zeros((len(self.input_data.index),), dtype=np.int32)
        self.sample_prop = pd.DataFrame({'dt': dt, 'real': real, 'role': role},
                                        index=self.input_data.index)
        print("data loaded: ", self.input_data.shape)

    def load_y(self, res_type, y_ratio_thr=0.6):
        """
        Load labels

        :param res_type: str. Label type
        :param y_ratio_thr: float. Invalid column if valid ratio < y_ratio_thr
        """
        if res_type == "cct":
            self.y = pd.read_table(
                self.data_path + "/cct.txt", encoding='gbk', sep='\t', index_col=0)
            self.y[self.y <= 0.0] = np.NaN
            self.y[self.y >= 0.99] = np.NaN
        elif res_type == "sst":
            self.y = pd.read_table(
                self.data_path + "/sst.txt", encoding='gbk', sep='\t', index_col=0)
            # y[(y['al']<0.14)|(y['al']>0.22)] = np.NaN
            # y[(y['f']<0.57)|(y['f']>0.68)] = np.NaN
        elif res_type == "vs":
            self.y = pd.read_table(
                self.data_path + "/vs.txt", encoding='gbk', sep='\t', index_col=0)
        elif res_type == "stv":
            self.y = pd.read_table(
                self.data_path + "/stv.txt", encoding='gbk', sep='\t', index_col=0)
            self.y[self.y <= 0.0] = np.NaN
            self.y[self.y >= 0.99] = np.NaN
        self.y = self.y.reindex(self.input_data.index)
        valid = np.sum(self.y.isna() == False, axis=0)
        valid = (valid / self.y.shape[0]) > y_ratio_thr
        self.y = self.y.loc[:, valid.values]
        print("y loaded: ", self.y.shape)

    def get_y_indices(self, targets):
        """
        Get indices of labels

        :param targets: [names].
        :return: [indices].
        """
        ret = []
        for t in targets:
            ret.extend(np.where(self.y.columns == t)[0])
        ret = sorted(set(ret))
        return ret

    def dt_index(self, dt_begin, dt_end=None):
        """
        Get DateTime indices

        :param dt_begin: DataTime.
        :param dt_end: DataTime or None. DataTime for range, None for one
        :return: Index.
        """
        if dt_end is not None:
            return self.sample_prop.index[
                self.sample_prop.dt.between(dt_begin, dt_end)]
        return self.sample_prop.index[self.sample_prop.dt == dt_begin]

    def get_column_minmax(self):
        """
        Initial column_vali, column_min, column_max before normalize
        """
        delta_thres = {'gen_p': 0.1,
                       'gen_u': 0.001,
                       'st_pg': 0.1,
                       'st_pl': 0.1,
                       'st_ql': 0.1,
                       'dc_p': 0.1,
                       'dc_q': 0.1,
                       'dc_acu': 0.001,
                       'ed': 0.00001}

        self.column_min = np.nanmin(self.input_data, axis=0)
        self.column_max = np.nanmax(self.input_data, axis=0)
        max_min = self.column_max - self.column_min

        elem_info = load_elem_info(self.model_path + "/elem_info.dat")
        gens = elem_info[elem_info['type'] == 5]
        for i, (t, elem) in enumerate(self.input_layer):
            if t == 'gen_p':
                self.column_min[i] = gens['limit1'][elem]
                self.column_max[i] = gens['limit2'][elem]
            elif t == 'st_pg':
                sub_gens = gens[gens.station == elem]
                self.column_min[i] = np.sum(sub_gens['limit1'])
                self.column_max[i] = np.sum(sub_gens['limit2'])
            elif t == 'gen_u' or t == 'dc_acu':
                self.column_min[i] = 0.9
                self.column_max[i] = 1.15
            # elif t=='st_pl' or t=='st_ql' or t=='ed' or t=='dc_p' or
            # t=='dc_q':
            else:
                d80 = (self.column_max[i] - self.column_min[i]) / 8
                self.column_min[i] -= d80
                self.column_max[i] += d80
            self.column_valid[i] = self.column_valid[i] \
                & (np.isnan(max_min[i]) == False) \
                & (max_min[i] > delta_thres[t])

    def normalize(self, range_min=-1.0, range_max=1.0):
        """
        Normalize use column_min and column_max

        :param range_min: float. lower limit of feature range.
        :param range_max: float. Upper limit of feature range.
        """
        self.get_column_minmax()
        column_min = self.column_min.reshape((1, len(self.column_min)))
        column_max = self.column_max.reshape((1, len(self.column_max)))
        deltas = column_max - column_min + 1e-8
        data = (range_max - range_min) * \
               (self.input_data.values - column_min) / deltas + range_min
        self.input_data = pd.DataFrame(data=data,
                                       index=self.input_data.index,
                                       columns=self.input_data.columns)
        ed_nas = self.input_data.isna() & self.input_data.columns.str.startswith("ed")
        self.input_data[ed_nas] = range_max
        self.input_data.fillna(value=range_min, inplace=True)
        self.input_data.clip(lower=range_min, upper=range_max, inplace=True)

    def drop_times(self, times):
        """
        Drop samples with times

        :param times: [times]. Index, the same as dir name of data
        """
        self.input_data.drop(index=times, inplace=True)
        self.sample_prop.drop(index=times, inplace=True)
        self.y.drop(index=times, inplace=True)

    def drop_labels(self, labels):
        """
        Drop lables

        :param labels: [label].
        """
        self.y.drop(columns=labels, inplace=True)

    def split_dataset_dt(self, dt_train_begin, dt_train_end,
                         dt_test_begin=None, dt_test_end=None,
                         val_ratio=0.1):
        """
        Split dataset by DateTime, act on sample_prop.role, -1 for not used

        :param dt_train_begin: DateTime.
        :param dt_train_end: DateTime.
        :param dt_test_begin: DateTime.
        :param dt_test_end: DateTime.
        :param val_ratio: float.
        :return: [ids]. Valid data set IDs.
        """
        ids = [0]
        self.sample_prop.loc[self.sample_prop.dt.between(dt_train_begin, dt_train_end),
                             'role'] = 0
        if 0.0 < val_ratio <= 1.0:
            val_bool = (self.sample_prop.role == 0)
            val_indices = np.where(val_bool)[0]
            np.random.shuffle(val_indices)
            val_indices = val_indices[int(len(val_indices) * val_ratio):]
            val_bool[val_indices] = False
            self.sample_prop.loc[val_bool, 'role'] = max(ids) + 1
            ids.append(max(ids) + 1)
        if dt_test_begin is not None:
            if dt_test_end is None:
                dt_test_end = dt_test_begin \
                    + datetime.timedelta(days=1, seconds=-1)
            self.sample_prop.loc[self.sample_prop.dt.between(dt_test_begin,
                                                             dt_test_end),
                                 'role'] = max(ids) + 1
            ids.append(max(ids) + 1)
        return ids

    def split_dataset_random(self, ratios=[1.0]):
        """
        Split dataset randomly, act on sample_prop.role, -1 for not used

        :param ratios: [float]. The ratios of every sub set.
        :return: [ids]. Valid data set IDs.
        """
        sum_ratios = sum(ratios) + 1e-8
        indices = np.array(range(self.input_data.shape[0]))
        np.random.shuffle(indices)
        role_column = np.where(self.sample_prop.columns == 'role')[0]
        self.sample_prop.role = -1
        start = 0
        ids = []
        for i, r in enumerate(ratios):
            num = int(self.input_data.shape[0] * r / sum_ratios)
            self.sample_prop.iloc[indices[start:start + num], role_column] = i
            start += num
            ids.append(i)
        return ids

    def make_dataset_tensor(self, y_column,
                            n_batch=16, n_shuffle=1000,
                            only_real=False):
        """
        Make data set from tensor slices.

        :param y_column: int. Index of label
        :param n_batch: int
        :param n_shuffle: int
        :param only_real: bool
        :return: [(dataset, n_sample, n_feature)].
        """
        data_sets = []
        roles = list(set(self.sample_prop.role))
        for r in roles:
            if r < 0:
                continue
            row_valid = (self.sample_prop.role == r).values
            y = self.y.values[:, y_column:y_column + 1]
            row_valid = row_valid & np.any(np.isnan(y) == False, axis=1)
            if only_real:
                row_valid = row_valid & self.sample_prop.real
            data = self.input_data.loc[row_valid, self.column_valid].values
            y = y[row_valid, :]
            dataset = tf.data.Dataset.from_tensor_slices((data, y))
            dataset = dataset.batch(n_batch)
            dataset = dataset.shuffle(n_shuffle)
            dataset = dataset.repeat()
            print("make dataset [%d]" % r, data.shape, y.shape)
            data_sets.append((dataset, data.shape[0], data.shape[1]))
        return data_sets

    def make_dataset_tensors(self, y_columns=None, only_real=False):
        """
        Make dataset for self.data_sets, use for generator

        :param y_columns: [columns].
        :param only_real: bool.
        """
        if not y_columns:
            y_columns = list(range(self.y.shape[1]))
        roles = list(set(self.sample_prop.role))
        for r in roles:
            if r < 0:
                continue
            y = self.y.values[:, y_columns]
            row_valid = (self.sample_prop.role == r).values
            row_valid = row_valid & np.any(np.isnan(y) == False, axis=1)
            if only_real:
                row_valid = row_valid & self.sample_prop.real
            if not np.any(row_valid):
                print("make dataset [%d] is empty" % r)
                continue
            data = self.input_data.loc[row_valid, self.column_valid].values
            y = y[row_valid]
            if len(y_columns) == 1:
                y = y.reshape(y.shape[0], 1)
            self.data_sets[r] = (data, y, self.sample_prop.index[row_valid])
            row_size = np.sum(row_valid)
            column_size = np.sum(self.column_valid)
            sample_size = np.sum(np.isnan(y) == False)
            print("make dataset [%d]" % r,
                  (row_size, column_size, sample_size))

    def dataset_generator(self, data, labels, n_batch,
                          is_shuffle=True):
        """
        Dataset generator for 1 label once

        :param data: 2D np.array.
        :param labels: 2D np.array.
        :param n_batch: int.
        :param is_shuffle: bool.
        :yield: ({'x': xs, 'fault': fs}, {'y': ys})
        """
        sample_size = np.sum(np.isnan(labels) == False)
        # print("dataset_generator: data =", datas.shape,
        #   "label =", labels.shape,
        #   "n_batch =", n_batch,
        #   "sample_size =", sample_size)
        assert sample_size >= n_batch
        batch_indices = np.zeros((n_batch, 2), dtype=np.int32)
        # round_indices = np.zeros((sample_size, 2), dtype=np.int32)
        round_indices = np.vstack(np.where(not np.isnan(labels))).T
        if is_shuffle:
            indices = np.arange(sample_size)
            np.random.shuffle(indices)
            round_indices = round_indices[indices]
        curr = 0
        while True:
            batch_begin = curr
            batch_end = batch_begin + n_batch
            if batch_end > sample_size:
                batch_indices[:sample_size - batch_begin, :] =\
                    round_indices[batch_begin:sample_size, :]
                batch_indices[sample_size - batch_begin:, :] =\
                    round_indices[0:batch_end - sample_size, :]
            else:
                batch_indices[:] = round_indices[batch_begin:batch_end, :]
            xs = data[batch_indices[:, 0], :]
            fs = batch_indices[:, 1].copy()
            fs = fs.reshape((n_batch, 1))
            ys = labels[batch_indices[:, 0], batch_indices[:, 1]]
            ys = ys.reshape((n_batch, 1))
            yield ({'x': xs, 'fault': fs}, {'y_': ys})
            curr += n_batch
            if curr >= sample_size:
                curr = 0
                if is_shuffle:
                    indices = np.arange(sample_size)
                    np.random.shuffle(indices)
                    round_indices = round_indices[indices]

    def dataset_generator_multi(self, datas, labels, n_batch,
                                is_shuffle=True):
        """
        Dataset generator for multiple labels, 1 sample(n labels) once

        :param datas: 2D np.array.
        :param labels: 2D np.array.
        :param n_batch: int.
        :param is_shuffle: bool.
        :yield: ({'x': xs, 'fault': fs, 'y_': ys}, {})
        """
        # print("dataset_generator: data =", datas.shape,
        #   "label =", labels.shape,
        #   "n_batch =", n_batch,
        #   "sample_size =", datas.shape[0])
        round_indices = np.arange(datas.shape[0])
        curr = 0
        while True:
            if curr + n_batch > datas.shape[0]:
                curr = 0
                if is_shuffle:
                    np.random.shuffle(round_indices)
                continue
            else:
                batch_indices = round_indices[curr:curr + n_batch]
            xs = datas[batch_indices, :].copy()
            ys = labels[batch_indices, :].copy()
            ys[np.isnan(ys)] = 0.0
            fs = (ys > 0.0)
            fs = fs.astype(np.float32)
            yield ({'x': xs, 'fault': fs, 'y_': ys}, {})
            curr += n_batch

    def get_dataset(self, id=1):
        """
        Get dataset

        :param role: id
        :return: tuple(data, labels, indices)
        """
        return self.data_sets.get(id, (None, None, None))


if __name__ == '__main__':

    path = "/home/sdy/python/db/2018_11"
    if os.name == 'nt':
        path = "d:/python/db/2018_11"
    input_dic = {'gen': ['gen_p', 'gen_u'],
                 'st': ['st_pg', 'st_pl', 'st_ql'],
                 'dc': ['dc_p', 'dc_q', 'dc_acu'],
                 'ed': ['ed']}

    net = GHNet("inf", input_dic)
    net.load_net(path + "/net")

    data_set = GHData(path,
                      path + "/net",
                      net.input_layer)
    t = time.time()
    data_set.load_x()
    data_set.load_y('cct')
    data_set.normalize()
    """
    data_set.split_dataset_dt(dt_train_begin=datetime.datetime(2018, 11, 1),
                              dt_train_end=datetime.datetime(
                                  2018, 11, 21) - datetime.timedelta(seconds=1),
                              dt_test_begin=datetime.datetime(2018, 11, 21),
                              dt_test_end=datetime.datetime(
                                  2018, 11, 22) - datetime.timedelta(seconds=1),
                              validation_perc=0.5)
    """
    data_set.split_dataset_random([0.8, 0.1, 0.1])
    data_set.make_dataset_tensors()
    gen = data_set.dataset_generator_multi(data_set.data_sets[1][0],
                                           data_set.data_sets[1][1],
                                           n_batch=32)
    it = iter(gen)
    inputs, _ = next(it)
    print("runtime = ", time.time() - t)
