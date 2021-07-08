import tensorflow as tf

'''
单个Reader,多个样本
'''

# 生成一个先入先出队列和一个QueueRunner,生成文件名队列
filenames = ['./demo_data/A.csv', './demo_data/B.csv', './demo_data/C.csv', './demo_data/C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=True, capacity=3, num_epochs=None)
print(type(filename_queue))

# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
print(type(key), type(value))

# 定义Decoder
# example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

# 这句话加上，就会不打乱顺序
# example_batch, label_batch = tf.train.shuffle_batch(
#     [example, label], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)


# example_batch, label_batch = tf.train.batch(
#       [example, label], batch_size=5)

# 运行Graph
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队



    # e_val, l_val = sess.run([example, label])
    # print(e_val, l_val)

    for i in range(100):
        k, v = sess.run([key, value])
        print(i, k, v)
        # example.eval(), label.eval()

        # e_val, l_val = sess.run([example_batch, label_batch])
        # print(e_val, l_val)
        # print(example_batch.eval(), label_batch.eval())

        # a, b = [example_batch.eval(), label_batch.eval()]
        # print(a, b)


    coord.request_stop()
    coord.join(threads)
