# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn

Active learning
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import backend as K
import datetime
import os
import sys

from model.ghnet_model import GHNet
from model.ghnet_util import dataset_predict
from data.ghnet_data import GHData


if __name__ == '__main__':

    path = "/home/sdy/python/db/2018_11"
    if os.name == 'nt':
        path = "d:/python/db/2018_11"

    all_types = ['cct', 'sst', 'vs', 'v_curve']
    res_type = 'cct'
    res_name = res_type
    res_path = path + "/" + res_name
    input_dic = {'gen': ['gen_p', 'gen_u'],
                 'st': ['st_pl', 'st_ql']}

    dr_percs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    net_path = path + "/net"
    net = GHNet("inf", input_dic, dr_percs=dr_percs)
    net.load_net(net_path)
    net1 = GHNet("inf", input_dic, dr_percs=dr_percs)
    net1.load_net(net_path)

    data_set = GHData(path,
                      net_path,
                      net.input_layer)
    data_set.load_x(x_ratio_thr=-1.0)
    data_set.load_y(res_type)
    data_set.normalize()
    drops = np.array(range(len(data_set.column_valid)))[~data_set.column_valid]
    net.drop_inputs(drops)
    net1.drop_inputs(drops)

    n_batch = 16
    n_epochs = 10
    n_al_epochs = 10
    only_real = True
    y_columns = list(range(data_set.y.shape[1]))
    y_columns = [2, 11, 23]
    net.build_multi_reg_k(len(y_columns),
                          activation=tf.keras.layers.LeakyReLU())
    net1.build_multi_reg_k(len(y_columns),
                           activation=tf.keras.layers.LeakyReLU())
    data_set.split_dataset_random(train_perc=0.0,
                                  val_perc=0.1, notused_perc=0.9)
    data_set.make_dataset_tensors(y_columns=y_columns,
                                  only_real=only_real)
    val_sample_size = data_set.val_labels.shape[0]
    validation_steps = val_sample_size // n_batch
    val_gen = data_set.dataset_generator_multi(
        data_set.val_datas, data_set.val_labels, n_batch)

    for al in range(n_al_epochs):
        al_res = dataset_predict(net.pre_model, data_set, role=0)[:, 1]
        al_res = al_res.reshape((-1, len(y_columns)))
        al_res1 = dataset_predict(net1.pre_model, data_set, role=0)[:, 1]
        al_res1 = al_res1.reshape((-1, len(y_columns)))
        delta = np.abs(al_res - al_res1).mean(axis=1)
        al_as = delta.argsort()[::-1]
        new_train_num = min(np.sum(delta > 0.02), 200)
        idx = data_set.notused_indices[al_as[:new_train_num]]
        data_set.sample_prop.role[idx] = 1
        print("add %d new train samples." % new_train_num)
        data_set.make_dataset_tensors(y_columns=y_columns,
                                      only_real=only_real)

        train_sample_size = data_set.train_labels.shape[0]
        steps_per_epoch = train_sample_size // n_batch
        train_gen = data_set.dataset_generator_multi(
            data_set.train_datas, data_set.train_labels, n_batch)
        net.model.fit_generator(train_gen,
                                epochs=n_epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=validation_steps)
        net1.model.fit_generator(train_gen,
                                 epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=val_gen,
                                 validation_steps=validation_steps)
        val_res = dataset_predict(net.pre_model, data_set, role=2)
        val_res = val_res[np.isnan(val_res[:,0])==False]
        print("val error = ", np.abs(val_res[:, 2]).mean())
