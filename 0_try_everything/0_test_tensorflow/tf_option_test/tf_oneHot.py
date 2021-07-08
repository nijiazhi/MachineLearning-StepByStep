import tensorflow as tf
import numpy as np


# Truth = np.array([0, 0, 1, 0])
Truth = np.array([-1, 0.0, 1.0, 4.0, 1.0])
Truth_onehot = tf.one_hot(Truth, 3)


with tf.Session() as sess:
    print(sess.run(tf.argmax(Truth)))
    print(sess.run(Truth_onehot))


