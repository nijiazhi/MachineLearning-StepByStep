import tensorflow as tf
import numpy as np


'''
正则化常用到集合，下面是最原始的添加正则办法
（直接在变量声明后将之添加进'losses'集合或tf.GraphKeys.LOESSES也行）
'''
def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)
cur_lay = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weights = get_weights([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_lay = tf.nn.relu(tf.matmul(cur_lay, weights) + bias)
    in_dimension = layer_dimension[i]

mess_loss = tf.reduce_mean(tf.square(y_ - cur_lay))
tf.add_to_collection('losses', mess_loss)

loss_collection = tf.get_collection('losses')
print(loss_collection)

loss = tf.add_n(loss_collection)
print(loss)