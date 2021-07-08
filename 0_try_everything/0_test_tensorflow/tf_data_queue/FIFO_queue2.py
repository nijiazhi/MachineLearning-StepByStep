# coding:utf-8

import tensorflow as tf

'''
使用队列进行训练
'''

# train_queue = tf.FIFOQueue(
#     capacity=10, dtypes=[tf.string, tf.int32],
#     shapes=[[3], [3]], shared_name=None, name=None
# )
#
# files = [['11', '21', '31'], ['10', '20', '30'], ['12', '22', '32'], ['12', '22', '32']]
# label = [(1, 2, 1), (2, 1, 2), (1, 1, 1), (1, 1, 2)]


train_queue = tf.FIFOQueue(
    capacity=10, dtypes=[tf.string, tf.int32],
    shapes=[(2, 3), 3], shared_name=None, name='andyni'
)

# 2行3列的输入，对应shape中的(2, 3)
files = [[['11', '21', '31'], ['11', '21', '31']], [['11', '21', '31'], ['10', '20', '30']]]

# 长度为3的输入，对应shape中的[3]
label = [(1, 2, 1), (2, 1, 2)]


# 一个入队列操作
train_enqueue = train_queue.enqueue_many([files, label])



with tf.Session() as sess:

    # 队列初始化
    sess.run(train_enqueue)

    # 队列元素
    print(sess.run(train_queue.size()))
    print()

    # q = sess.run(train_queue)
    # print(q)



