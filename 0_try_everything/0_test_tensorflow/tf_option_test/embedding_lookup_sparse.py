import numpy as np
import pandas as pd
import tensorflow as tf


a = tf.Variable(np.arange(8).reshape(2, 4))
b = tf.Variable(np.arange(100, 108).reshape(2, 4))
c = tf.Variable(np.arange(1000, 1008).reshape(2, 4))


idx = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 2, 0], dense_shape=(2, 3))
# embedded_tensor = tf.nn.embedding_lookup_sparse((a, b, c), idx, None, combiner='mean')
# embedded_tensor = tf.nn.embedding_lookup_sparse([a, b, c], idx, None, combiner='sum')
embedded_tensor = tf.nn.embedding_lookup_sparse([a, b, c], idx, None)


'''
    len(params)>1时候，会启用partition_strategy策略，两种模式：div mod

    两种模式的操作参数都为len(params)

    具体策略与params中元素【第一维长度之和】有关，这里面是2+1+2=5
'''


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

    print("########## sparse matrix #########")
    print(sess.run(idx))

    print("########## embedded_tensor #########")
    print(sess.run(embedded_tensor))
