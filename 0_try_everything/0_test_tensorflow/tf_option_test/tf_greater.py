import tensorflow as tf

s1 = [1, 2, 3]
v1 = tf.Variable(s1)


def test():
    s2 = [1, 2, 3]
    v2 = tf.Variable(s2)
    return v2

a = tf.Variable(1, name="a", collections=[tf.GraphKeys.LOCAL_VARIABLES])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    test()

    # s3 = [1, 2, 3]
    # v3 = tf.Variable(s3)

    global_vals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print(global_vals)

    local_vals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    print(local_vals)

    # VARIABLES 和 GLOBAL_VARIABLES一样
    vals = tf.get_collection(tf.GraphKeys.VARIABLES)
    print(vals)


    # 正题，tf中的 比较函数 和 where函数
    print(tf.where(tf.greater(10, 2), 'Yes', 'No').eval())
