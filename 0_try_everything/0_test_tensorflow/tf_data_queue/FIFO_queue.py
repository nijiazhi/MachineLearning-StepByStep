# coding:utf-8

'''
【队列与多线程】

FIFOQueue类基于基类QueueBase．QueueBase主要包含入列（enqueue）和出列（dequeue）两个操作．
enqueue操作返回计算图中的一个Operation节点，dequeue操作返回一个Tensor值．
Tensor在创建时同样只是一个定义，需要放在Session中运行才能获得真正的数值．

tensorflow使用tf.FIFOQueue类创建一个先入先出队列．
属性：
capacity：指定队列中的元素数量的上限。
dtypes：DType对象的列表。dtypes的长度必须等于每个队列元素中张量的数量。
shapes：（可选项）
names:(可选项）命名队列的字符串。
shared_name :(可选项）如果非空，则将在多个会话中以给定名称共享此队列。
name：队列操作的可选名称。
'''


import tensorflow as tf

# 创建一个先入先出的队列,指定队列最多可以保存3个元素,并指定类型为整数
q = tf.FIFOQueue(3, 'int32')


# 初始化队列中的元素,将[0,10,20] 3个元素排入此队列
# queue_init = q.enqueue_many(([0, 10, 30],))
queue_init = q.enqueue_many([[0, 10, 30]])


# 将队列中的第1个元素出队列,并存入变量x中
x = q.dequeue()

# 将得到的值加1
y = x + 1

# 将加1后的值重新加入队列
q_inc = q.enqueue([y])


'''
这里要注意的是：如果是两个向量，它们是无法调用

tf.concat(1, [t1, t2])
来连接的，因为它们对应的shape只有一个维度，当然不能在第二维上连了，虽然实际中两个向量可以在行上连，但是放在程序里是会报错的
如果要连，必须要调用tf.expand_dims来扩维
'''
# feature_image_list = [[[1, 0]], [[2, 0]], [[3, 0]]]  # 此数组中装有待拼接的内部元素，其shape为(1,2)
# tf_concat = tf.concat(feature_image_list, 0)

with tf.Session() as sess:

    # 队列初始化
    print(q.size())
    queue_init.run()
    print(q.size())
    print()

    print(sess.run(q.size()))
    print()

    for _ in range(5):
        # 执行数据出队列/出队元素+1/重新加入队列的过程
        v, _ = sess.run([x, q_inc])
        print(v)


    # v = sess.run(tf_concat)
    # print(v, ' | ', len(v))

'''
队列开始有[0,10,20]三个元素，执行5次数据出队列，出队元素+1，重新加入队列的过程中：
x=0, 　y=1,　　 队列：[10，20，1]
x=10,　　y=11,　　队列：[20，1，11]
x=20,　　y=21,　　队列：[1，11，21]
x=1,　 　y=2,　　 队列：[11，21，2]
x=11,　　y=12,　　队列：[21，2，12]
'''
