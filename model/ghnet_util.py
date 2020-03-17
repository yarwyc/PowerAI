# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def write_input(data_set, file_name):
    """
    Write input infos to file, including type, elem, min, max

    :param data_set: GHData. Use its input_layer and column_*.
    :param file_name: str.
    """
    with open(file_name, "w") as f:
        for i, (t, elem) in enumerate(data_set.input_layer):
            if data_set.column_valid[i]:
                f.write("%s %s %f %f\n" % (t, elem,
                                           data_set.column_min[i],
                                           data_set.column_max[i]))


def write_output(targets, file_name):
    """
    Write output (predictions) to file.

    :param targets: iterable.
    :param file_name: str.
    """
    with open(file_name, "w") as f:
        f.write('\n'.join(targets))


def write_adjust(adjusts, file_name):
    """
    Write adjust infos to file.

    :param adjusts: {elem: (value, delta)}.
    :param file_name: str.
    :return:
    """
    with open(file_name, "w") as f:
        for k, (v, dv) in adjusts.items():
            f.write("%s %.4f %.4f\n" % (k, v, dv))


def save_model(path, name, model, suffix='h5'):
    """
    Save model for .h5/.json/.pb

    :param path: str.
    :param name: str.
    :param model: tf.keras.models.Model.
    :param suffix: str. For different format.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if suffix == 'h5':
        model.save(path + "/" + name + ".h5")
    elif suffix == 'json':
        model_json = model.to_json()
        with open(path + "/" + name + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path + "/" + name + ".h5")
    elif suffix == 'pb':
        builder = tf.saved_model.builder.SavedModelBuilder(path + "/model")
        sess = K.get_session()
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING])
        builder.add_meta_graph(["GHNet"])
        builder.save()
        # sess = K.get_session()
        # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        #                       sess,
        #                       sess.graph_def,
        #                       output_node_names=["y/BiasAdd"])
        # tf.train.write_graph(frozen_graph_def,
        #                       path,
        #                       name + '.pb',
        #                       as_text = False)


def load_model(path, name, suffix='h5'):
    """
    Load model from .h5/.json/.pb

    :param path: str.
    :param name: str.
    :param suffix: str. For different format.
    :return: tf.keras.models.Model.
    """
    model = None
    if suffix == 'h5':
        model = tf.keras.models.load_model(path + "/" + name + ".h5")
    elif suffix == 'json':
        json_file = open(path + "/" + name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(path + "/" + name + ".h5")
    elif suffix == 'pb':
        pass
    return model


def dataset_predict(model, data_set, role):
    """
    Predict for data_set.

    :param model: tf.keras.models.Model. Pre_model.
    :param data_set: GHData.
    :param role: int. Data set id.
    :return: tuple (labels, predictions). Labels and predictions of the same shape.
    :raise: ValueError.
    """
    (data, labels, _) = data_set.get_dataset(role)
    if data is None or labels is None:
        raise ValueError('no data')
    pre = model.predict(data)
    return labels, pre


def save_data_plt(data, path, y=None):
    """
    Save curve for each column of data and y, [data[i], y]

    :param data: DataFrame.
    :param path: str.
    :param y: 1D np.array.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(data.shape[1]):
        plt.plot(np.arange(data.shape[0]), data.iloc[:, i])
        if y is not None:
            plt.plot(np.arange(y.shape[0]), y)
        # plt.ylim((-1, 1))
        plt.title(data.columns[i], fontproperties="SimHei")
        plt.savefig(path + "/%d.jpg" % i)
        plt.close()


def plt_loss(history):
    """
    Plot loss curve.

    :param history: Keras history infos for training.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


if __name__ == '__main__':
    pass
