#-*- coding:utf-8 -*-

'''
多个reader，多个样本
'''
import tensorflow as tf

filenames = ['./demo_data/A.csv', './demo_data/B.csv', './demo_data/C.csv']
# filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

#num_epoch: 设置迭代数
filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=3)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [['null'], ['null']]

# 定义了多种解码器, 每个解码器跟一个reader相连
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]  # Reader设置为2

# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
example_batch, label_batch = tf.train.batch_join(
      example_list, batch_size=5)

init_local_op = tf.initialize_local_variables()
with tf.Session() as sess:
    sess.run(init_local_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            e_val, l_val = sess.run([example_batch, label_batch])
            print(e_val, l_val)
    except tf.errors.OutOfRangeError:
        print('Epochs Complete!')
    finally:
        coord.request_stop()

    # for i in range(10):
    #     e_val,l_val = sess.run([example_batch,label_batch])
    #     print(e_val,l_val)

    coord.join(threads)
    coord.request_stop()

