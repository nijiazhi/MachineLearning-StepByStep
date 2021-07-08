import tensorflow as tf
import numpy as np


# Truth = np.array([0 ,0, 1, 0])
# Pred_logits = np.array([3.5, 2.1, 7.89, 4.4])


# Truth = np.array([0.0, 1.0, 0.0])
# Truth_onehot = tf.one_hot(Truth, 2)
# Pred_logits = np.array([0, 0.4, 56.1])

Truth = np.array([0.0, 1.0])
Truth_int = tf.to_int32(Truth)

test_Truth = np.array([[0.0, 0.0], [0.0, 1.0]])  # 注意这里的内容，如果shape不匹配，会自动填充重负的值
Truth_onehot = tf.one_hot(Truth, 2)

Pred_logits = np.array([[0.4, 0.8], [0.9, 0.3]])
sigmoid_Pred_logits = np.array([0.4, -0.6])


loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=test_Truth, logits=Pred_logits)
loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Truth_onehot, logits=Pred_logits)
loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Truth_onehot), logits=Pred_logits)
# loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Truth_int, logits=Pred_logits)

loss4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=Truth, logits=sigmoid_Pred_logits)


##################

# softmax 使用时，logits必须两维度（相当于sigmoid的一维），不然都为1，无意义了
# softmax = tf.nn.softmax(Pred_logits, name='softmax')
# sigmoid = tf.nn.sigmoid(Pred_logits, name='sigmoid')

# 下面两者的结果，应该是一致的，相减之后可以得到相应的结果
softmax = tf.nn.softmax(np.array([0.6, 1]), name='softmax')
sigmoid = tf.nn.sigmoid(np.array([0.4]), name='sigmoid')


with tf.Session() as sess:
    print(sess.run(Truth_int))
    print(sess.run(tf.argmax(Truth)))
    print(sess.run(Truth_onehot))
    print('\n', '*' * 60, '\n')

    print(sess.run(loss1), '\n')
    print(sess.run(loss2), '\n')
    print(sess.run(loss3), '\n')
    print(sess.run(loss4), '\n')
    print('\n', '*'*60, '\n')

    print(sess.run(softmax), '\n')
    print(sess.run(sigmoid), '\n')


