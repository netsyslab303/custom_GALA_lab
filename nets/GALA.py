import tensorflow as tf
import torch
from tensorflow import keras


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

    def Laplacian_smoothing(self, x, name, training, DAD):
        tmp = []
        inputs = self.GALA[name](x, training=training)
        for i in range(len(x)):
            tmp.append(tf.nn.relu(
                tf.sparse.sparse_dense_matmul(DAD, inputs[i])))
        tmp = tf.convert_to_tensor(tmp)
        return tmp

    def Laplacian_sharpening(self, x, name, training, DAD):
        tmp = []
        for i in range(len(x)):
            tmp.append(tf.nn.relu(tf.sparse.sparse_dense_matmul(DAD, self.GALA[name](x[i], training=training))))
        return tmp

    def call(self, H, training=None):
        for i in range(3):
            if i < 2:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.s_DADsm)
            else:
                H = self.Laplacian_smoothing(H, 'enc%d' % i, training, self.t_DADsm)
        self.H = H
        for i in range(3):
            if i == 0:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.t_DADsp)
            else:
                H = self.Laplacian_sharpening(H, 'dec%d' % i, training, self.s_DADsp)
        return H
