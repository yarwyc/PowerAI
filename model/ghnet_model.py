# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from core.power import load_elem_info, load_station_info


def slice_k(x, begin, end):
    return x[:, begin:end]


class MultiRegLossLayer(layers.Layer):
    """
    Custom loss layer with multiple labels considering invalid label.
    """

    def __init__(self, **kwargs):
        super(MultiRegLossLayer, self).__init__(**kwargs)

    def multi_loss(self, y, y_, fault):
        """
        Loss formula

        :param y: 2d np.array. Batch of predicting y.
        :param y_: 2d np.array. Batch of real y.
        :param fault: 2d np.array. Batch of valid fault mask.
        :return: float. Loss value.
        """
        loss = layers.multiply([(y - y_), fault])
        return K.sum(K.abs(loss)) / K.sum(fault)

    def call(self, inputs):
        """
        Callable method

        :param inputs: [y, y_, fault]
        :return:
        """
        y, y_, fault = inputs
        loss = self.multi_loss(y, y_, fault)
        self.add_loss(loss, inputs=inputs)
        return y


class GHNet(object):
    """
    GHNet class, from gdnet.dat to tf model

    Attributes:
        name: str. Net name
        input_types: dict{elem_type: [value_type]}. Valid inputs.
        exclude_areas: [areas]. Invalid areas.
        decay_ratios: [ratios]. Decay ratios for each layer, n_out / n_in = ratio.
        nodes: [[nodes1], [nodes2], ...]. Nodes of each layer, from lower to upper.
        input_layer: [(type, name)]. The same as GHData
        ghnet: DataFrame. Infos in ghnet.dat.
        st_info: DataFrame. Station Info.
        elem_info: DataFrame. Element Info.
        gens: DataFrame. Generator Info.
        ed_info: DataFrame. Electric distance Info.
        W_init: Initial method of W

        x: tf.keras.Input. Input of static state quantities.
        fault: tf.keras.Input. Input of fault mask.
        x_feature: {name: tf.Tensor}. Dict of features of different layers.
        y: tf.Tensor. Output of model.
        y_: tf.keras.Input. Input of real label.
        train_model: tf.keras.models.Model. Model for training.
        pre_model: tf.keras.models.Model. Model for predicting.
        gradients: [tf.Tensor]. Gradients for each y to x.
    """

    def __init__(self, name, input_types, exclude_areas=None,
                 decay_ratios=None):
        """
        Initial method

        :param name: str. Net name
        :param input_types: dict{elem_type: [value_type]}. Valid inputs.
        :param exclude_areas: [areas]. Invalid areas.
        :param decay_ratios: [ratios]. Decay ratios for each layer, n_out / n_in = ratio.
        """
        self.name = name
        self.input_types = input_types
        self.exclude_areas = exclude_areas or []
        self.decay_ratios = decay_ratios or [0.5] * 8
        self.nodes = []
        self.input_layer = []
        self.ghnet = None
        self.st_info = None
        self.elem_info = None
        self.gens = None
        self.ed_info = None
        self.W_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        # self.W_init = 'glorot_uniform'

        self.x = None
        self.fault = None
        self.x_feature = {}
        self.y = None
        self.y_ = None
        self.train_model = None
        self.pre_model = None
        self.gradients = None

    def build_inf(self, activation=layers.LeakyReLU()):
        """
        Build inference model according to ghnet.dat, not including head

        :param activation: activation function.
        :return: Output of ghnet.
        """
        x_size = len(self.input_layer)
        print("build_inf begin: input_size =", x_size)
        self.x = Input(shape=(x_size,), dtype='float32', name='x')
        x_upper = {}
        for layer, nodes in enumerate(self.nodes):
            x_lower = x_upper.copy()
            x_upper = {}
            for node in nodes:
                x_sub = []
                if node.in_end - node.in_begin > 0:
                    x_input = Lambda(slice_k, output_shape=(node.in_end - node.in_begin,),
                                     arguments={'begin': node.in_begin, 'end': node.in_end})(self.x)
                    x_sub.append(x_input)
                x_sub.extend([x_lower[sub_name]
                              for sub_name in node.subnets if sub_name in x_lower])
                x_sub = x_sub[0] if len(x_sub) == 1 else \
                    layers.concatenate(x_sub, axis=-1)
                if x_sub == [] or x_sub.shape[1] <= 0:
                    print("node(%s) has no input, drop it" % node.name)
                    continue
                node.set_io_num(int(x_sub.shape[-1]),
                                dr=self.decay_ratios[layer],
                                n_layer=2)
                for n_hidden in node.hiddens:
                    x_sub = layers.Dense(n_hidden,
                                         kernel_initializer=self.W_init)(x_sub)
                    x_sub = activation(x_sub)
                x_upper[node.name] = x_sub
                # print("node(%s) added, n_in=%d, n_out=%d"%(node.name,node.n_in, node.n_out))
        x_out = [sub_net for sub_net in x_upper.values()]
        if not x_out:
            raise RuntimeError("x_out is empty")
        self.x_feature['sub_out'] = x_out[0] if len(x_out) == 1 else \
            layers.concatenate(x_out, axis=-1)
        print("build_inf end: out_size =", self.x_feature['sub_out'].shape)
        return self.x_feature['sub_out']

    def build_reg(self, n_fault, activation=layers.LeakyReLU(),
                  output_activation=None):
        """
        Build reg model from input to y, include build_inf

        :param n_fault: int. Amount of faults.
        :param activation:
        :param output_activation:
        :return:
        """
        print("build_reg begin: n_fault =", n_fault)
        x = self.build_inf(activation)
        if n_fault > 1:
            self.fault = Input(shape=(1,), dtype='int32', name='fault')
            fault = layers.Embedding(
                n_fault, 5, name='fault_embedding')(self.fault)
            fault = layers.Flatten()(fault)
            x = layers.concatenate([x, fault], axis=-1)
        # x = layers.Dense(64, kernel_initializer=self.W_init)(x)
        # x = activation(x)
        x = layers.Dense(32, kernel_initializer=self.W_init)(x)
        x = activation(x)
        self.y = layers.Dense(1, activation=output_activation,
                              kernel_initializer=self.W_init, name='y')(x)
        if n_fault > 1:
            self.train_model = Model([self.x, self.fault], self.y)
        else:
            self.train_model = Model(self.x, self.y)
        self.pre_model = self.train_model
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                           loss='mae',
                           metrics=['mae'])
        print("build_reg end!")

    def build_multi_reg(self, n_fault, activation=layers.LeakyReLU(),
                        output_activation=None):
        """
        Build multiple reg model from input to y, include build_inf

        :param n_fault: int. Amount of faults.
        :param activation:
        :param output_activation:
        :return:
        """
        print("build_multi_reg begin: n_fault =", n_fault)
        x = self.build_inf(activation)
        # x = layers.Dense(64, kernel_initializer=self.W_init)(x)
        # x = activation(x)
        x = layers.Dense(32, kernel_initializer=self.W_init)(x)
        x = activation(x)
        self.x_feature['high_out'] = x
        self.y = layers.Dense(n_fault, activation=output_activation,
                              kernel_initializer=self.W_init, name='y')(x)
        self.pre_model = Model(self.x, self.y)
        self.gradients = []
        for i in range(n_fault):
            self.gradients.append(K.gradients(self.y[:, i], [self.x]))
        self.fault = Input(shape=(n_fault,), dtype='float32', name='fault')
        self.y_ = Input(shape=(n_fault,), dtype='float32', name='y_')
        loss = MultiRegLossLayer(name='multi_loss_layer')([self.y, self.y_, self.fault])
        self.train_model = Model([self.x, self.fault, self.y_], loss)
        self.train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                                 loss=None)
        print("build_multi_reg end!")

    def get_station_input(self, st_name):
        """
        Get station inputs.

        :param st_name: str. Station name.
        :return: [(type, name)].
        """
        # if self.gens is None or self.st_info is None or self.ed_info is None:
        # 	raise RuntimeError("power information not loaded")
        if st_name not in self.st_info.index:
            return []
        ret = []
        st = self.st_info.loc[st_name]
        if 'ed' in self.input_types:
            ed_info = self.ed_info[self.ed_info['j'] == st_name]
            for _, ed in ed_info.iterrows():
                ret.append(('ed', '%s_%s' % (ed['i'], ed['j'])))
        if 'generator' in self.input_types:
            if st.type == 50:
                for gen in self.gens[self.gens['station'] == st_name].index:
                    ret.extend([('generator_' + it, gen)
                                for it in self.input_types['generator']])
        if 'station' in self.input_types:
            if st.type == 50:
                ret.extend(
                    [('station_' + it, st_name) for it in self.input_types['station']
                     if it in {'pg', 'qg', 'sg'}])
            elif 51 <= st.type <= 52:
                ret.extend(
                    [('station_' + it, st_name) for it in self.input_types['station']
                     if it in {'pg', 'pl'}])
            elif st.type == 60:
                ret.extend(
                    [('station_' + it, st_name) for it in self.input_types['station']
                     if it in {'pl', 'ql'}])
            # else:
            # 	ret.extend([('station_' + it,st_name)
            #               for it in self.input_types['station']])
        if 'dcline' in self.input_types:
            if 41 <= st.type <= 44:
                ret.extend([('dcline_' + it, st_name)
                            for it in self.input_types['dcline']])
        return ret

    def load_net(self, path):
        """
        Load ghnet.dat

        :param path: str. Contains ghnet.dat, st_info.dat, elem_info.dat.
        :return:
        """
        self.ghnet = pd.read_table(path + '/ghnet.dat', encoding='gbk', sep=' ')
        self.ghnet.sort_values(by='layer', inplace=True)
        self.st_info = load_station_info(path + '/st_info.dat')
        self.elem_info = load_elem_info(path + '/elem_info.dat')
        self.gens = self.elem_info[self.elem_info['type'] == 5]
        self.ed_info = pd.read_table(path + '/ed_info.dat', encoding='gbk', sep=' ')

        # input stations
        self.nodes = []
        used_input_names = set()
        upper_names = []
        start = 0
        input_layer_no = np.min(self.ghnet['layer'])
        for layer in range(input_layer_no, np.max(self.ghnet['layer']) + 1):
            lower_names = upper_names.copy()
            upper_names = []
            used_subnet_names = []
            nodes = []
            for _, sub in self.ghnet[self.ghnet['layer'] == layer].iterrows():
                if sub['upper'] in self.st_info.index and \
                        self.st_info.loc[sub['upper']]['area'] in self.exclude_areas:
                    continue
                ss = sub['lower'].split('+')
                # ss.append(sub['upper'])
                # ss = list(set(ss))
                node = Node(layer, sub['upper'])
                if layer == input_layer_no or layer == input_layer_no + 1:  # 220kV and 500kV
                    for name in ss:
                        ret = self.get_station_input(name)
                        self.input_layer.extend(ret)
                        if name in used_input_names:
                            print("station(%s) has been used as input" % name)
                        else:
                            used_input_names.add(name)
                    node.in_begin = start
                    node.in_end = len(self.input_layer)
                    # node.inputs = list(range(node.in_begin, node.in_end))
                    start = node.in_end
                for name in ss:
                    if name in lower_names:
                        lower_node_name = node.make_name(layer - 1, name)
                        lower_node = self.get_node(lower_node_name)
                        if lower_node is not None:
                            node.subnets.append(lower_node_name)
                            used_subnet_names.append(name)
                if node.in_begin == node.in_end and len(node.subnets) == 0:
                    print("node(%s) has no input, drop it" % node.name)
                    continue
                nodes.append(node)
                upper_names.append(sub['upper'])
            not_used_lower = set(lower_names) - set(used_subnet_names)
            if not_used_lower:
                print("not used lower node: " + '+'.join(not_used_lower))
                # raise NotImplementedError('%d node(s) are not connected'%len(not_used_lower))
            self.nodes.append(nodes)
        print("load ghnet successfully")

    def print_node(self, node, fp=sys.stdout):
        """
        Print node infos.

        :param node: Node.
        :param fp: fd.
        """
        fp.write("...." * (len(self.nodes) - node.layer) + "%s(%d,%d)\n" %
                 (node.name, node.in_end - node.in_begin, len(node.subnets)))
        for i in range(node.in_begin, node.in_end):
            fp.write("...." * (len(self.nodes) - node.layer + 1) +
                     ",".join(self.input_layer[i]) + "\n"),
        for sub in node.subnets:
            self.print_node(self.get_node(sub), fp)

    def get_node(self, name):
        """
        Get node by name

        :param name: str. Node name.
        :return: node or None
        """
        for nodes in self.nodes:
            for node in nodes:
                if (node.name == name):
                    return node
        return None

    def drop_inputs(self, drops):
        """
        Drop inputs by indices.

        :param drops: 1D np.array. Drop indices.
        """
        for nodes in self.nodes:
            for node in nodes:
                if node.in_begin < 0:
                    continue
                node.in_begin -= np.sum(drops <= node.in_begin)
                if node.in_begin < 0:
                    node.in_begin = 0
                node.in_end -= np.sum(drops < node.in_end)
                # node.inputs = list(range(node.in_begin, node.in_end))
        self.input_layer = [x for i, x in enumerate(self.input_layer)
                            if i not in drops]


class Node:
    """
    Node class used by GHNet.

    Attributes:
        layer: int. Number of layer.
        name: str. Node name.
        in_begin: int. Begin index of inputs in the lower layer.
        in_end: int. End index of inputs in the lower layer.
        subnets: [names]. Names of sub nets.
        n_in: int. Amount of input.
        n_out: int. Amount of output.
        hiddens: [int]. Numbers of neuron of hidden layers.
    """

    def __init__(self, layer, name):
        self.layer = layer
        self.name = self.make_name(layer, name)
        # self.inputs = []
        # index of original inputs
        self.in_begin = -1
        self.in_end = -1
        self.subnets = []
        # index of sub nets
        self.n_in = 0
        self.n_out = 0
        self.hiddens = []

    def __repr__(self):
        return "name:%s n_in:%d n_out:%d" \
               % (self.name, self.n_in, self.n_out)

    def make_name(self, layer, name):
        return "%d_%s" % (layer, name)

    def set_io_num(self, n_in, n_out=0, dr=0.5, n_layer=1):
        """
        Set input and output size, as well as hiddens.

        :param n_in: int. Amount of input.
        :param n_out: int. Amount of output.
        :param dr: float. Ratio of decay.
        :param n_layer:
        :raise RuntimeError
        """
        self.n_in = n_in
        if self.n_in <= 0:
            raise RuntimeError("node(%s) n_in<=0" % self.name)
        self.n_out = int(self.n_in * dr) if n_out <= 0 else n_out
        if self.n_out <= 0:
            self.n_out = 1
        n_reduce = int((self.n_in - self.n_out) / n_layer)
        self.hiddens = [self.n_in - n_reduce * i for i in range(1, n_layer)]
        self.hiddens.append(self.n_out)


if __name__ == '__main__':

    path = "/home/sdy/python/db/2018_11"
    if os.name == 'nt':
        path = "d:/python/db/2018_11"
    input_dic = {'generator': ['p', 'v'],
                 'station': ['pg', 'pl', 'ql'],
                 'dcline': ['p', 'q', 'acu'],
                 'ed': ['ed']}
    net = GHNet("inf", input_dic)
    net.load_net(path + "/net")
    with open(path + "/net/ghnet_out.txt", "w") as f:
        net.print_node(net.nodes[-1][0], f)
    # net.drop_inputs(np.array(range(320)))
    net.build_inf()
    net.build_multi_reg(2)
