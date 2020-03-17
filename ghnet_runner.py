# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import sys
import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile

from model.ghnet_model import GHNet
from data.ghnet_data import GHData
from model.ghnet_util import dataset_predict, save_model, write_input, write_output, write_adjust
from common.time_util import timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--res_type', default='cct',
                        help='Result type (cct, sst, vs, stv)')
    parser.add_argument('--task_type', default='train',
                        help='Task type (train, test, lmd, feature, match, adjust, app_pb)')
    parser.add_argument('--path', default=None,
                        help='Path to data (*.npz), net_path=path/net, res_path=path/res_type')
    parser.add_argument('--dt_begin', default='2018_11_27T10_00_00',
                        help='E format DateTime (yyyy_mm_ddTHH_MM_SS) for begin')
    parser.add_argument('--dt_end', default='2018_11_30T23_59_00',
                        help='E format DateTime (yyyy_mm_ddTHH_MM_SS) for end')
    parser.add_argument('--dataset_split', default='random',
                        help='Data set split strategy (random, dt)')
    args = parser.parse_args()

    path = "d:/python/db/2018_11" if args.path is None else args.path
    net_path = path + "/net"
    res_path = path + "/" + args.res_type
    input_dic = {'gen': ['gen_p', 'gen_u'],
                 'st': ['st_pl', 'st_ql']}
    '''
    input_dic = {'gen':['gen_p','gen_u'],
                 'st':['st_pg', 'st_pl','st_ql'],
                 'dc':['dc_p','dc_q','dc_acu'],
                 'ed':['ed']}
    '''

    decay_ratios = [0.5] * 8
    net = GHNet("inf", input_dic, decay_ratios=decay_ratios)
    net.load_net(net_path)
    # with open(net_path+"/ghnet_out.txt", "w") as f:
    # 	net.print_node(net.nodes[-1][0], f)

    data_set = GHData(path, net_path, net.input_layer)
    data_set.load_x(x_ratio_thr=-1.0)
    data_set.load_y(args.res_type)
    data_set.normalize()
    drops = np.array(range(len(data_set.column_valid)))[~data_set.column_valid]
    net.drop_inputs(drops)
    y_columns = list(range(data_set.y.shape[1]))
    if args.res_type == 'cct':
        targets = ['东北.青北一线', '东北.燕董一线', '东北.丰徐二线']
        y_columns = data_set.get_y_indices(targets)
    elif args.res_type == 'sst':
        y_columns = [0]
    column_names = data_set.y.columns[y_columns]
    print("targets:", column_names)
    net.build_multi_reg(len(y_columns), activation=tf.keras.layers.LeakyReLU())

    n_batch = 16
    n_epochs = 10
    only_real = True
    if args.task_type == 'train':
        if args.dataset_split == 'random':
            ids = data_set.split_dataset_random(ratios=[0.9, 0.05, 0.05])
        else:
            dt_train_begin = datetime.datetime(2018, 8, 1)
            dt_train_end = datetime.datetime(2018, 8, 26) - datetime.timedelta(seconds=1)
            dt_test_begin = datetime.datetime(2018, 8, 26)
            dt_test_end = datetime.datetime(2018, 9, 1) - datetime.timedelta(seconds=1)
            ids = data_set.split_dataset_dt(dt_train_begin=dt_train_begin,
                                            dt_train_end=dt_train_end,
                                            dt_test_begin=dt_test_begin,
                                            dt_test_end=dt_test_end,
                                            val_perc=0.1)
        assert (0 in ids) and (1 in ids)
        data_set.make_dataset_tensors(y_columns=y_columns, only_real=only_real)
        train_data, train_labels, _ = data_set.get_dataset(0)
        train_sample_size = train_labels.shape[0]
        steps_per_epoch = train_sample_size // n_batch
        val_data, val_labels, _ = data_set.get_dataset(1)
        val_sample_size = val_labels.shape[0]
        validation_steps = val_sample_size // n_batch
        train_gen = data_set.dataset_generator_multi(train_data, train_labels, n_batch)
        val_gen = data_set.dataset_generator_multi(val_data, val_labels, n_batch)
        dt_run_start = datetime.datetime.now()
        with timer("Timer training"):
            history = net.train_model.fit_generator(train_gen,
                                                    epochs=n_epochs,
                                                    steps_per_epoch=steps_per_epoch,
                                                    validation_data=val_gen,
                                                    validation_steps=validation_steps)
        test_labels, pre = dataset_predict(net.pre_model, data_set, role=2)
        save_model(res_path, args.res_type, net.pre_model, suffix='pb')
        # save_model(res_path, args.res_type, net.pre_model, suffix='json')
        write_input(data_set, res_path + "/input.txt")
        write_output(data_set.y.columns[y_columns], res_path + "/output.txt")
        sys.exit(0)

    net.pre_model.load_weights(res_path + "/" + args.res_type + ".h5")
    dt_begin = datetime.datetime.strptime(args.dt_begin, "%Y_%m_%dT%H_%M_%S")
    dt_end = datetime.datetime.strptime(args.dt_end, "%Y_%m_%dT%H_%M_%S") \
             + datetime.timedelta(seconds=1)
    y = data_set.y.values[:, y_columns]
    row_valid = data_set.sample_prop.dt.between(dt_begin, dt_end).values
    if only_real:
        row_valid = row_valid & data_set.sample_prop.real
    if not np.any(row_valid):
        raise ValueError("empty dataset!")
    data = data_set.input_data.loc[row_valid, data_set.column_valid].values
    y = y[row_valid]
    if len(y_columns) == 1:
        y = y.reshape(y.shape[0], 1)

    if args.task_type == 'test':
        indices = data_set.sample_prop[row_valid].index
        with timer("Timer test"):
            pre = net.pre_model.predict(data)
        for i, name in enumerate(column_names):
            print("%s --> real=%f, pre=%f" % (name, y[0][i], pre[0][i]))
    elif args.task_type == 'lmd':
        sess = K.get_session()
        with timer("Timer lmd"):
            grad_ret = sess.run(net.gradients, feed_dict={net.x: data})
        max_min = data_set.column_max - data_set.column_min
        max_min = max_min[data_set.column_valid]
        input_layer = []
        for i, il in enumerate(data_set.input_layer):
            if data_set.column_valid[i]:
                input_layer.append(il)
        lmd_types = ['gen_p']
        for y_i in range(len(grad_ret)):
            lmd = grad_ret[y_i][0] / max_min
            # lmd = grad_ret[y_i][0]
            lmd = lmd.reshape((lmd.shape[1],))
            lmd_as = lmd.argsort()
            print("%s=%s, lmd=%f" % (
                input_layer[lmd_as[0]][0], input_layer[lmd_as[0]][1], lmd[lmd_as[0]]))
            print("%s=%s, lmd=%f" % (
                input_layer[lmd_as[-1]][0], input_layer[lmd_as[-1]][1], lmd[lmd_as[-1]]))
            with open(res_path + "/lmd_" + str(y_i) + ".res", "w") as f:
                for i in lmd_as:
                    if input_layer[i][0] in lmd_types:
                        f.write("%s=%s, lmd=%f\n" %
                                (input_layer[i][0], input_layer[i][1], lmd[i]))
                        # print("%s=%s, lmd=%f"%(input_layer[i][0], input_layer[i][1], lmd[i]))
    elif args.task_type == 'feature':
        sess = K.get_session()
        with timer("Timer feature"):
            features = sess.run(net.x_feature['high_out'], feed_dict={net.x: data})
        print(features)
    elif args.task_type == 'match':
        sess = K.get_session()
        with timer("Timer match"):
            x_features = sess.run(net.x_feature, feed_dict={net.x: data})
            data = data_set.input_data.values[:, data_set.column_valid]
            all_features = sess.run(net.x_feature, feed_dict={net.x: data})
            distances = (all_features - x_features)
            distances = np.sum(distances * distances, axis=1)
            dis_df = pd.Series(data=distances, index=data_set.input_data.index)
            dis_df = dis_df[data_set.sample_prop.real]
            for i, (idx, dis) in enumerate(dis_df.sort_values()[:10].items()):
                print("i = %d, dt = %s, dis = %f" % (i+1, idx, dis))
    elif args.task_type == 'adjust':
        cur_data = data_set.ori_data.loc[args.dt_begin].values[data_set.column_valid]
        max_data = data_set.column_max[data_set.column_valid]
        min_data = data_set.column_min[data_set.column_valid]
        max_min = max_data - min_data
        if data.shape[0] > 1:
            data = data[0:1, :]
        n_round = 17
        n_count = 10
        lmds = []
        adjusted_indices = set()
        power_per_round = -0.1  # 10MW, positive for promote stability
        y_i = 0
        lmd_types = ['gen_p']
        input_layer = []
        for i, il in enumerate(data_set.input_layer):
            if data_set.column_valid[i]:
                input_layer.append(il)
        sess = K.get_session()
        adjusts = {}
        features = []
        for rd in range(1, n_round + 1):
            # print("round %d" % rd)
            for k in adjusts.keys():
                adjusts[k] = (adjusts[k][0], 0.)
            pre0, grads = sess.run([net.y, net.gradients],
                                   feed_dict={net.x: data})
            pre0 = pre0[0][y_i]
            lmd = grads[y_i][0] * 2 / max_min
            lmd = lmd.reshape((lmd.shape[1],))
            lmds.append(lmd)
            lmd_as = lmd.argsort()
            valid_round = False
            file_name = "%s/round_%04d.glf" % (res_path, rd)
            count = 0
            for i in lmd_as:
                if input_layer[i][0] not in lmd_types \
                        or lmd[i] >= 0.0 \
                        or np.isnan(cur_data[i]) \
                        or cur_data[i] - power_per_round < min_data[i] or cur_data[i] - power_per_round > max_data[i]:
                    continue
                # print("gen=%s(%.2f-%.2f), lmd=%.6f, ori=%.2fMW, adjust= -10MW"\
                # 	%(input_layer[i][1], min_data[i]*100, max_data[i]*100, lmd[i], cur_data[i]*100))
                cur_data[i] = cur_data[i] - power_per_round
                data[0, i] = data[0, i] - 2 * power_per_round / max_min[i]
                adjusts["%s %s" % (input_layer[i][0], input_layer[i][1])] = (
                    cur_data[i], -power_per_round)
                pre0 = pre0 - lmd[i] * power_per_round
                adjusted_indices.add(i)
                valid_round = True
                count = count + 1
                if count >= n_count:
                    break
            count = 0
            for i in lmd_as[::-1]:
                if input_layer[i][0] not in lmd_types \
                        or lmd[i] <= 0.0 \
                        or np.isnan(cur_data[i]) \
                        or cur_data[i] + power_per_round < min_data[i] or cur_data[i] + power_per_round > max_data[i]:
                    continue
                # print("gen=%s(%.2f-%.2f), lmd=%.6f, ori=%.2fMW, adjust= +10MW"\
                # 	%(input_layer[i][1], min_data[i]*100, max_data[i]*100, lmd[i], cur_data[i]*100))
                cur_data[i] = cur_data[i] + power_per_round
                data[0, i] = data[0, i] + 2 * power_per_round / max_min[i]
                adjusts["%s %s" % (input_layer[i][0], input_layer[i][1])] = (
                    cur_data[i], power_per_round)
                pre0 = pre0 + lmd[i] * power_per_round
                adjusted_indices.add(i)
                valid_round = True
                count = count + 1
                if count >= n_count:
                    break
            write_adjust(adjusts, file_name)
            if not valid_round:
                print("no more adjust......")
            pre, feature = sess.run([net.y, net.x_feature['high_out']],
                                    feed_dict={net.x: data})
            features.append(feature)
            print("pre(lmd) = %f, pre(NN) = %f" % (pre0, pre[0][y_i]))
    elif args.task_type == 'app_pb':
        with timer("Timer app_pb"):
            sess = K.get_session()
            file_name = path + '/' + args.res_type + '/' + args.res_type + '.pb'
            with gfile.FastGFile(file_name, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read().strip())
                # graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                input_x = sess.graph.get_tensor_by_name('x:0')
                input_fault = sess.graph.get_tensor_by_name('fault:0')
                input_y_ = sess.graph.get_tensor_by_name('y_:0')
                output_y = sess.graph.get_tensor_by_name('y/BiasAdd:0')
                print(sess.run(output_y, feed_dict={
                    input_x: data, input_fault: np.array([[1.0, 0.0, 0.0]])}))
                print(sess.run(output_y, feed_dict={
                    input_x: data, input_fault: np.array([[0.0, 1.0, 0.0]])}))
                print(sess.run(output_y, feed_dict={
                    input_x: data, input_fault: np.array([[0.0, 0.0, 1.0]])}))
                print(sess.run(output_y, feed_dict={
                    input_x: data, input_fault: np.array([[0.0, 0.0, 0.0]])}))
