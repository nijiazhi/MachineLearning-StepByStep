# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

'''
[1]
在enqueue_many参数设置为False（默认值）的时候，
tf.train.batch的输出，是batch_size*tensor.shape，
其含义就是将tensors参数看做一个样本，那么batch_size个样本，只是单一样本的复制。
在其实际应用中，tensors参数一般对应的是一个文件，那么这样操作意味着从文件中读取batch_size次， 以获得一个batch的样本。

[2]
在enqueu_many参数设置为True的时候，
tf.train.batch将tensors参数看做一个batch的样本，
那么batch_size只是调整一个batch中样本的维度的，
因为输出的维度是batch_size*tensor.shape[1:]
（可以尝试将代码例子中batch_size改成3，再看结果）

[3]
最后需要注意的tf.train.batch的num_threads参数，指定的是进行入队操作的线程数量，
可以进行多线程对于同一个文件进行操作，这样会比单一线程读取文件快。

[4]
tf.train.batch_join一般就对应于多个文件的多线程读取，
可以看到当enqueue_many参数设置为False（默认值）的时候，tensor_list中每个tensor被看做单个样本，
这个时候组成batch_size的一个batch的样本，是从各个单一样本中凑成一个batch_size的样本。
可以看到由于是多线程，每次取值不同，
也就是类似，每个tensor对应一个文件，也对应一个线程，那么组成batch的时候，该线程读取到文件（例子中是tensor的哪个值）是不确定，
这样就形成了打乱的形成样本的时候。

[5]
而在enqueue_many参数设置为True的时候，
取一个batch的数据，是在tensor_list中随机取一个，
因为每一个就是一个batch的数据，batch_size只是截断或填充这个batch的大小。

[5]
tf.train.batch和tf.train.batch_join的区别，
一般来说，单一文件多线程，那么选用tf.train.batch（需要打乱样本，有对应的tf.train.shuffle_batch）；
而对于多线程多文件的情况，一般选用tf.train.batch_join来获取样本（打乱样本同样也有对应的tf.train.shuffle_batch_join使用）。
'''

# 5*4
# tensor_list = [ [1, 2,  3, 4], [2, 3, 4, 5] ]
tensor_list = ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20])

# 5*1*4
tensor_list2 = [ [[[1, 2, 3, 4]], [1]], [[[5, 6, 7, 8]], [2]] ]
# tensor_list2 = ([[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]], [[13, 14, 15, 16]], [[17, 18, 19, 20]])
# tensor_list2 = ([ [[[0, 0], [1, 2]]], [2] ], [ [[[0, 0], [1, 8]]], [8] ])
# tensor_list2 = ([ [2], [3] ], [ [8], [7] ])


x1 = tf.train.batch(tensor_list, batch_size=3, enqueue_many=False)
x2 = tf.train.batch(tensor_list, batch_size=3, enqueue_many=True)

y1 = tf.train.batch_join(tensor_list2, batch_size=3, enqueue_many=False)
y2 = tf.train.batch_join(tensor_list2, batch_size=4, enqueue_many=True)
print(y2)

with tf.Session() as sess:

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("\n\nx1 batch:" + "-" * 10)
    x1_batch = sess.run(x1)
    print(x1_batch)
    x1_m = np.array(tensor_list)
    x1_batch_m = np.array(x1_batch)
    print(x1_m.shape, x1_batch_m.shape)


    print("\n\nx2 batch:" + "-" * 10)
    x2_batch = sess.run(x2)
    print(x2_batch)
    x2_m = np.array(tensor_list)
    x2_batch_m = np.array(x2_batch)
    print(x2_m.shape, x2_batch_m.shape)


    print("\n\ny1 batch:" + "-" * 10)
    y1_batch = sess.run(y1)
    print(y1_batch)
    y1_m = np.array(tensor_list2)
    # y1_batch_m = np.array(y1_batch)
    # print(y1_m.shape, y1_batch_m.shape)
    print(y1_m.shape)


    print("\n\ny2 batch:" + "-" * 10)
    y2_batch = sess.run(y2)
    print(y2_batch)
    y2_m = np.array(tensor_list2)
    # y2_batch_m = np.array(y2_batch)
    # print(y2_m.shape, y2_batch_m.shape)

    print('\n\n', "-" * 10)
    coord.request_stop()
    coord.join(threads)
    print('All Done!')