# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# 根据[网络参数]和[输入]， 计算出[估计输出值y_hat]
def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.

  Tips:
    tf.truncated_normal：代表从3倍标准差的正态分布中随机取值（http://blog.csdn.net/u013713117/article/details/65446361）

  """

  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases

  return logits  # 返回最后一层的结果


# 定义loss函数，使用inference结果和真值计算最后的误差
def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.

  Tips：
    计算交叉熵时候有点小坑需要注意（http://blog.csdn.net/hejunqing14/article/details/52397824）
    第一个坑:logits表示从最后一个隐藏层线性变换输出的结果！假设类别数目为10，那么对于每个样本这个logits应该是个10维的向量，
             且没有经过归一化，所有这个向量的元素和不为1。然后这个函数会先将logits进行softmax归一化，
             然后与label表示的onehot向量比较，计算交叉熵。 也就是说，这个函数执行了三步（这里意思一下）：
             sm=nn.softmax(logits)
             onehot=tf.sparse_to_dense(label, …)
             nn.sparse_cross_entropy(sm,onehot)

    第二个坑:输入的label是稀疏表示的，就是是一个[0，10）的一个整数，这我们都知道。
            但是这个数必须是一维的！就是说，每个样本的期望类别只有一个，属于A类就不能属于其他类了。
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


# 训练函数，优化器来优化loss
def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard（可以方便可视化的显示loss变化）.

  Creates an optimizer and applies the gradients to all trainable variables.（使用优化器进行优化）

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.（使用sess.run启动整个训练）

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss（将loss加入summarizer，用于记录）
  tf.scalar_summary(loss.op.name, loss)

  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


# 评估函数，利用得到的模型
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES). 就是一个数字表示哪类

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  Tips:
        in_top_k函数用于在预测的时候判断预测结果是否正确。函数本来的意义为判断label是不是在logits 的前k大的值，返回一个布尔值。
        这个label是个index而不是向量.如果表示第4类为正确的，label不长[0,0,0,1,0]这样，
        而是label=3。所以如果你的label是onehot的，恭喜你，你要转化一下：
        labels = tf.argmax(y_, 1)
        topFiver = tf.nn.in_top_k(y, labels, 5) #in top 5
        第二个坑：同样的，label只有一个值，只能用于单类别判断！
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)  # 这个函数可以判断给出的预测，是否在前k个中
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))  # 返回正确值的数量
