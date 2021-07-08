import tensorflow as tf


sess=tf.Session()
weight_decay=0.1                                                #(1)定义weight_decay
l2_reg=tf.contrib.layers.l2_regularizer(weight_decay)           #(2)定义l2_regularizer()
tmp=tf.constant([0,1,2,3],dtype=tf.float32)
a=tf.get_variable("I_am_a",regularizer=l2_reg,initializer=tmp)  #(3)创建variable，l2_regularizer复制给regularizer参数


# 目测REXXX_LOSSES集合
# regularizer定义会将a加入REGULARIZATION_LOSSES集合
print("Global Set:")
keys = tf.get_collection("variables")
for key in keys:
  print(key.name)

print("Regular Set:")
keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
for key in keys:
  print(key.name)
print("--------------------")


sess.run(tf.global_variables_initializer())
print(sess.run(a))
reg_set=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)   #(4)则REGULARIAZTION_LOSSES集合会包含所有被weight_decay后的参数和，将其相加
l2_loss=tf.add_n(reg_set)
print("loss=%s" %(sess.run(l2_loss)))
