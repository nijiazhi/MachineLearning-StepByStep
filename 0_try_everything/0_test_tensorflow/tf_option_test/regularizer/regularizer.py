import tensorflow as tf


'''
L1和L2正则：

举例说明，假设有一个数组 nums=[1,2,3,4]

L1 = a*(|1|+|2|+|3|+|4|)

L2 = a*(1^2+2^2+3^2+4^2)/2

其中a是系数，用于平衡正则项与经验损失函数的权重关系，即：C = loss+a*Normalization。

'''

alpha = 0.5  # 系数设置为0.5，alpha相当于上述a

val = tf.constant([[1,2,3,4]],dtype=tf.float32)

l1 = tf.contrib.layers.l1_regularizer(alpha)(val)
l2 = tf.contrib.layers.l2_regularizer(alpha)(val)

tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2)
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
print('andy test:\t', type(reg_loss), reg_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("l1:", sess.run(l1))
    print("l2:", sess.run(l2))
