# coding:utf-8

import tensorflow as tf

w = tf.Variable(tf.zeros([2, 1]), name='weights')
b = tf.Variable(0., name='bias')


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


# part1：inference是给定x得到对应的预测y值
def inference(x):
    return tf.matmul(x, w)+b


# part2：loss是计算y和y_predict的差距
def loss(x, y):
    y_predicted = inference(x)
    return tf.reduce_sum(tf.squared_difference(y, y_predicted))

# part3：train是调用tensorflow提供的优化函数 对loss进行优化【输入是loss结果】
def train(total_loss):
    learning_rate = 1e-8
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess):
    print(sess.run(inference([[80., 25.]])))  # 303
    print(sess.run(inference([[65., 25.]])))  # 256


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)  # run 初始化
    train_x, train_y = inputs()
    for index in range(1000):
        sess.run(train(loss(train_x, train_y)))
        if index % 100 == 0:
            print(index)
    evaluate(sess)
