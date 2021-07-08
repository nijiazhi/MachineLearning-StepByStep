import numpy as np
import pandas as pd
import tensorflow as tf

# https://blog.csdn.net/u011974639/article/details/77647569


# partition_strategy='div' 的情况
a = tf.Variable(np.arange(8).reshape(2, 4))
b = tf.Variable(np.arange(8, 12).reshape(1, 4))
c = tf.Variable(np.arange(12, 20).reshape(2, 4))
# d = tf.Variable(np.arange(1, 9).reshape(2, 4))


embedded_tensor = tf.nn.embedding_lookup(params=[a, b, c], ids=[1, 4, 0], partition_strategy='div', name="embedding")
'''
    len(params)>1时候，会启用partition_strategy策略，两种模式：div mod
    具体策略与params中元素【第一维长度之和】有关，这里面是2+1+2=5
'''


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(a.get_shape())
    print(sess.run(b))
    print(sess.run(c))
    # print(sess.run(d))

    print("########## embedded_tensor #########")
    print(sess.run(embedded_tensor))
