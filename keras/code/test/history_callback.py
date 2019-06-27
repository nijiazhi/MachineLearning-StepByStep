from IPython.display import SVG
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import model_to_dot
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# 设置随机数种子,保证实验可重复
import numpy as np
np.random.seed(0)

# 设置线程
THREADS_NUM = 20
tf.ConfigProto(intra_op_parallelism_threads=THREADS_NUM)

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
print('原数据结构：')
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 数据变换：分为10个类别
nb_classes = 10

x_train_1 = X_train.reshape(60000, 784)
x_train_1 = x_train_1.astype('float32')
x_train_1 /= 255
y_train_1 = np_utils.to_categorical(Y_train, nb_classes)
print('变换后的数据结构：')
print(x_train_1.shape, y_train_1.shape)

x_test_1 = X_test.reshape(10000, 784)
y_test_1 = np_utils.to_categorical(Y_test, nb_classes)
print(x_test_1.shape, y_test_1.shape)

##############
## 搭建model
##############
model = Sequential()
model.add(Dense(nb_classes, input_shape=(784,)))  # 全连接，输入784维度, 输出10维度，需要和输入输出对应
model.add(Activation('softmax'))

sgd = SGD(lr=0.005)
# binary_crossentropy，就是交叉熵函数
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# model 概要
print(model.summary())


###########################
## 使用histroy callback
###########################
from keras.callbacks import Callback, TensorBoard


# 构建一个记录的loss的回调函数
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# 构建一个自定义的TensorBoard类，专门用来记录batch中的数据变化
class BatchTensorBoard(TensorBoard):

    def __init__(self, log_dir='./',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):

        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.batch = 0
        self.batch_queue = set()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.batch = self.batch + 1

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = float(value)
            summary_value.tag = "batch_" + name
            if (name, self.batch) in self.batch_queue:
                continue
            self.writer.add_summary(summary, self.batch)
            self.batch_queue.add((name, self.batch))
        self.writer.flush()


################
## 开始训练
################
tensorboard = TensorBoard(log_dir='../../log/test_log/epoch')
my_tensorboard = BatchTensorBoard(log_dir='../../log/test_log/batch')

history = model.fit(
    x_train_1[:500], y_train_1[:500],
    nb_epoch=5,
    verbose=1,
    batch_size=100,
    callbacks=[tensorboard, my_tensorboard],
)

print('\n\n', 'history:\n', history.history)