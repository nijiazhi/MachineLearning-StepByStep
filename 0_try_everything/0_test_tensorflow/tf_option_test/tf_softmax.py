
import numpy as np
import tensorflow as tf


w = np.array([[-0.48178506, -0.93234694],
 [ 0.63895214, -0.11406606],
 [ 0.30022776, -1.0633346 ],
 [-0.839464,    0.45388606],
 [-0.05854656 , 0.60846084],
 [-0.5509218,   0.3123361 ],
 [-0.20493148 ,-0.19308278],
 [-0.4055732 , -0.09931038],
 [ 0.21522444 ,-0.05413657],
 [-2.0957682 , -0.16523668],
 [-1.33988 ,  0.9419921]])

w1 = tf.Variable(w, dtype=tf.float32)
print('w:\t', w1)

b = [[-0.11460277, -0.11485916]]
b1 = tf.Variable(b)
print('b:\t', b1)
print()


x = [42.23, 12.45, 6.3, 42.23, 12.45, 6.3, 42.23, 12.45, 6.3, 42.23, 12.45]
x1 = tf.Variable(x)
x11 = tf.reshape(x1, shape=(1, -1))
print(x1, x11)

y = tf.matmul(x11, w1)+b1
softmax_y = tf.nn.softmax(y)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print(y, '\t', y.eval())
    print(softmax_y, '\t', softmax_y.eval())
    print('\n')


import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

w_ = w[:, 0] - w[:, 1]
b_ = b[0][0] - b[0][1]
w1 = np.array(w_).reshape((1, -1))
x1 = np.array(x).reshape((-1, 1))
print(w1.shape, x1.shape)
y = np.dot(w1, x1)+b_

print(y, '\t', sigmoid(y))
