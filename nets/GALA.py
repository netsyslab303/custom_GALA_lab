import tensorflow as tf
import numpy as np
import torch
import train_ws_seperated_temporal
from tensorflow import keras

use_dyna_adj = train_ws_seperated_temporal.use_dyna_adj
input_size = train_ws_seperated_temporal.input_size


class Model(tf.keras.Model):
    def __init__(self, s_DAD, t_DAD, name='GALA', batch_size=64, trainable=True, time_input=15, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        D = [64, 64, 128, 128, 256, 256]
        self.batch_size = batch_size
        self.GALA = {}
        self.model_len = len(D)
        inputs = keras.Input(shape=(17 * time_input, input_size,), name="digits")
        for i, d in enumerate(D):
            if i == 0:
                self.GALA['enc%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                self.GALA['enc%d' % i](inputs)
            else:
                self.GALA['enc%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                self.GALA['enc%d' % i](keras.Input(shape=(17 * time_input, D[i - 1],)))

        if trainable:
            for i, d in enumerate(D[self.model_len:0:-1] + [2]):
                self.GALA['dec%d' % i] = tf.keras.layers.Dense(d, trainable=trainable)  # , use_bias = False)
                if i == 0:
                    self.GALA['dec%d' % i](keras.Input(shape=(17 * time_input, D[-1],)))
                else:
                    self.GALA['dec%d' % i](keras.Input(shape=(17 * time_input, D[-i],)))

        org_score = keras.Input(shape=(17 * time_input, 1,), name="digits")
        self.GALA['dyna_s_adj'] = tf.keras.layers.Dense(1, trainable=trainable)
        self.GALA['dyna_s_adj'](org_score)
        self.GALA['dyna_t_adj'] = tf.keras.layers.Dense(1, trainable=trainable)
        self.GALA['dyna_t_adj'](org_score)

        self.s_DADsm = tf.sparse.SparseTensor(s_DAD['DADsm_indices'], s_DAD['DADsm_values'][0], s_DAD['dense_shape'][0])
        self.s_DADsp = tf.sparse.SparseTensor(s_DAD['DADsp_indices'], s_DAD['DADsp_values'][0], s_DAD['dense_shape'][0])
        self.t_DADsm = tf.sparse.SparseTensor(t_DAD['DADsm_indices'], t_DAD['DADsm_values'][0], t_DAD['dense_shape'][0])
        self.t_DADsp = tf.sparse.SparseTensor(t_DAD['DADsp_indices'], t_DAD['DADsp_values'][0], t_DAD['dense_shape'][0])

    def Laplacian_smoothing(self, x, name, training, DAD, dyna, dyna_zero):
        inputs = self.GALA[name](x, training=training)
        dad = tf.sparse.to_dense(DAD)
        if use_dyna_adj:
            dyna_inverse = 1-dyna_zero
            dyna_inverse = tf.multiply(dyna, dyna_inverse)
            dyna = tf.math.multiply(dyna, dad)
            dyna_zero = tf.math.multiply(dyna_zero, dad)
            dyna_inverse = tf.math.multiply(dyna_inverse, dad)
            sum_dad = tf.reduce_sum(dad, 1, keepdims=True)
            sum_dyna = tf.reduce_sum(dyna, 2, keepdims=True)
            ratio = sum_dad/sum_dyna
            dad_others = tf.math.multiply(dyna_zero, ratio) #
            dad_noised = tf.math.multiply(dyna_inverse, ratio)
            dad = dad_noised + dad_others
        output = tf.nn.relu(tf.matmul(dad, inputs))
        return output

    def Laplacian_sharpening(self, x, name, training, DAD, dyna, dyna_zero):
        inputs = self.GALA[name](x, training=training)
        dad = tf.sparse.to_dense(DAD)
        if use_dyna_adj:
            dyna_inverse = 1-dyna_zero
            dyna_inverse = tf.multiply(dyna, dyna_inverse)
            dyna = tf.math.multiply(dyna, dad)
            dyna_zero = tf.math.multiply(dyna_zero, dad)
            dyna_inverse = tf.math.multiply(dyna_inverse, dad)
            sum_dad = tf.reduce_sum(dad, 1, keepdims=True)
            sum_dyna = tf.reduce_sum(dyna, 2, keepdims=True)
            ratio = sum_dad/sum_dyna
            dad_others = tf.math.multiply(dyna_zero, ratio) #
            dad_noised = tf.math.multiply(dyna_inverse, ratio)
            dad = dad_noised + dad_others
        output = tf.nn.relu(tf.matmul(dad, inputs))
        return output

    def call(self, H, training=None, dyna=None, dyna_zero=None):
        tmp = tf.ones([dyna.shape[0], dyna.shape[1], 1])
        s_dyna = np.array(self.GALA['dyna_s_adj'](tmp, training=training))
        t_dyna = np.array(self.GALA['dyna_t_adj'](tmp, training=training))
        s_dyna = tf.multiply(s_dyna, 1 - dyna_zero) + dyna_zero
        t_dyna = tf.multiply(t_dyna, 1 - dyna_zero) + dyna_zero
        for i in range(self.model_len):
            if i % 2 == 0:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.s_DADsm, s_dyna, dyna_zero)
            else:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.t_DADsm, t_dyna, dyna_zero)
        self.H = H
        for i in range(self.model_len):
            if i % 2 == 0:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.t_DADsp, t_dyna, dyna_zero)
            else:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.s_DADsp, s_dyna, dyna_zero)
        return H
