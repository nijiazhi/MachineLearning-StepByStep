import tensorflow as tf

'''
可以看看这个说明

https://blog.csdn.net/u012436149/article/details/53696970
'''


with tf.variable_scope("level1") as scope:
    v1 = tf.get_variable("w", shape=[1])
    # scope = tf.get_variable_scope()  # 获取当前所在variable的score
    print(scope.name, '#', v1.name)

    with tf.variable_scope(scope, "level2", [1, 2]):
    # with tf.variable_scope("level2", [1, 2]):
    #     v2 = tf.get_variable("w", shape=[1])
        v2 = tf.get_variable("w1", shape=[1])
        print(scope.name, '#', v2.name)
print()

'''
# 嵌套的variable_scope会被reuse
# 如果这里【reuse=False】会报错
with tf.variable_scope("level1", reuse=None):
    # v3 = tf.get_variable("w", shape=[1])
    # scope = tf.get_variable_scope()
    # print(scope.name, '#', v3.name)

    v_test = tf.Variable(0, name="w")
    # v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
    # v_test = tf.get_variable("w", shape=[1])
    scope = tf.get_variable_scope()
    print(scope.name, '#', v_test.name)

    # with tf.variable_scope("level2", reuse=False):
    #     v4 = tf.get_variable("w", shape=[1])
    #     scope = tf.get_variable_scope()
    #     print(scope.name, v4.name, scope.reuse)
print()


with tf.variable_scope("level3"):
    v5 = tf.get_variable("w", shape=[1])
    scope = tf.get_variable_scope()
    print(scope.name, '#', v5.name, '#', v5)

scope = 'level4'
with tf.variable_scope(scope, reuse=None):
    v6 = tf.get_variable("w", shape=[1])
    print(v6.name, v6)


print('\n', '*'*30)
scope = tf.get_variable_scope()
print(scope, scope.name)
'''



with tf.variable_scope("andy_level") as scope:
    v1 = tf.Variable(name="w", initial_value=0)
    # scope = tf.get_variable_scope()  # 获取当前所在variable的score
    print(scope.name, '#', v1.name)

    with tf.variable_scope(scope, "andy_level2"):
        v2 = tf.Variable(name="w1", initial_value=0)
        print(scope.name, '#', v2.name)