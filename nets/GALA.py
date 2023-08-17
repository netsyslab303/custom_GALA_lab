import tensorflow as tf
import numpy as np
import torch
import train_ws_seperated_temporal
from tensorflow import keras

use_dyna_adj = train_ws_seperated_temporal.use_dyna_adj

class Model(tf.keras.Model):
    def __init__(self, s_DAD, t_DAD, name='GALA', batch_size=64, trainable=True, time_input=15, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.GALA = {}
        D = [400, 300, 100]
        inputs = keras.Input(shape=(17*time_input, 2,), name="digits")
        for i, d in enumerate(D):
            if i == 0:
                self.GALA['enc%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                self.GALA['enc%d' % i](inputs)
            else:
                self.GALA['enc%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                self.GALA['enc%d' % i](keras.Input(shape=(17*time_input, D[i - 1],)))

        if trainable:
            for i, d in enumerate(D[1::-1] + [2]):
                self.GALA['dec%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                self.GALA['dec%d' % i](keras.Input(shape=(17*time_input, D[-(i + 1)],)))

        self.s_DADsm = tf.sparse.SparseTensor(s_DAD['DADsm_indices'], s_DAD['DADsm_values'][0], s_DAD['dense_shape'][0])
        self.s_DADsp = tf.sparse.SparseTensor(s_DAD['DADsp_indices'], s_DAD['DADsp_values'][0], s_DAD['dense_shape'][0])
        self.t_DADsm = tf.sparse.SparseTensor(t_DAD['DADsm_indices'], t_DAD['DADsm_values'][0], t_DAD['dense_shape'][0])
        self.t_DADsp = tf.sparse.SparseTensor(t_DAD['DADsp_indices'], t_DAD['DADsp_values'][0], t_DAD['dense_shape'][0])

    def Laplacian_smoothing(self, x, name, training, DAD, dyna):
        inputs = self.GALA[name](x, training=training)
        dad = tf.sparse.to_dense(DAD)
        if use_dyna_adj:
            dad = tf.math.multiply(dyna, dad)
        output = tf.nn.relu(tf.matmul(dad,inputs))
        return output

    def Laplacian_sharpening(self, x, name, training, DAD, dyna):
        inputs = self.GALA[name](x, training=training)
        dad = tf.sparse.to_dense(DAD)
        if use_dyna_adj:
            dad = tf.math.multiply(dyna, dad)
        output = tf.nn.relu(tf.matmul(dad, inputs))
        return output

    def call(self, H, training=None, dyna=None):
        for i in range(3):
            if i < 2:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.t_DADsm, dyna)
            else:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.s_DADsm, dyna)
        self.H = H
        for i in range(3):
            if i == 0:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.s_DADsp, dyna)
            else:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.t_DADsp, dyna)
        return H
