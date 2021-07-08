# coding:utf8

import numpy as np
import tensorflow as tf


s1 = [1, 2]
s2 = [2, 4]

s = np.array(s1) - np.array(s2)
dist = np.sum(np.square(s))
print(dist)

v1 = tf.Variable(s1)
v2 = tf.Variable(s2)
print(v1, v2)

v11 = tf.reshape(v1, shape=(1, -1))
v22 = tf.reshape(v2, shape=(1, -1))
print(v11, v22)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    sub = tf.subtract(v1, v2)
    print(sub, sub.eval())

    square = tf.square(sub)
    print(square, square.eval())

    reduce_sum = tf.reduce_sum(square)
    print(reduce_sum, reduce_sum.eval())
