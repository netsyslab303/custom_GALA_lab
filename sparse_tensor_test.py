import tensorflow as tf
import numpy as np

def dense_to_sparse(dense):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)
    return sparse

a = np.array([[[1,1,1,1],[1,1,1,1]],[[1,1,1,1],[1,1,1,1]]],dtype='float32')
b = np.array([[[1,1,1,1]],[[1,1,1,1]]],dtype='float32')
b = np.transpose(b,(0,2,1))
print(a*b)
a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)
print(a)
print(b)

a = dense_to_sparse(a)
b = dense_to_sparse(b)

c = tf.sparse.sparse_dense_matmul(a,b)
print(c)