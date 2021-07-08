# coding : utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载 mnist数据集合
# 函数DataSet.next_batch()是用于获取以batch_size为大小的一个元组，
# 其中包含了一组图片和标签，该元组会被用于当前的TensorFlow运算会话中
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 构建数据占位符，用于图执行时候feed
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])

# 创建参数变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 得到预测值y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 使用 交叉熵 作为损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 还得自己写交叉熵，不开心。。。

# 训练过程，使用cross_entropy作为目标函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化所有vriables变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)  # run 初始化

# batch梯度下降
for i in range(1000):
  print('cur step:  ', i)
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # run 梯度下降

# 计算评测指标，准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # run 准确率