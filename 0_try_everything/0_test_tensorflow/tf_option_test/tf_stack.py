
import numpy as np
import tensorflow as tf

ph_feature_file = tf.placeholder(tf.string, shape=[5], name="ph_feature_file")
print(ph_feature_file)

s1 = [1, 2]
s2 = [3, 4]

v1 = tf.Variable(s1)
v2 = tf.Variable(s2)
print(v1, v2)
print()

v11 = tf.reshape(v1, shape=(1, -1))
v22 = tf.reshape(v2, shape=(1, -1))
print(v11, v22)
print()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    rs1 = tf.stack([v11, v22], axis=0)
    print(rs1, '\n', rs1.eval())
    print()

    rs2 = tf.unstack(v11, axis=0)
    print(rs2, '\n', sess.run(rs2))
    print()

    rs2 = tf.unstack(rs1, axis=1)
    print(rs2, '\n', sess.run(rs2))

