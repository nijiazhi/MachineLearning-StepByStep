# coding: utf-8
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


######################################
# TODO: set the gpu memory using fraction #
#####################################
def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# KTF.set_session(get_session(0.6))  # using 60% of total GPU Memory
# os.system("nvidia-smi")  # Execute the command (a string) in a subshell
# raw_input("Press Enter to continue...")
######################


batch_size = 128
nb_classes = 10
nb_epoch = 10
nb_data = 28 * 28
log_filepath = './'

# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# rescale

X_train = X_train.astype(np.float32)
X_train /= 255

X_test = X_test.astype(np.float32)
X_test /= 255
# convert class vectors to binary class matrices (one hot vectors)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(nb_data,), init='normal', name='dense1'))  # a sample is a row 28*28
model.add(Activation('relu', name='relu1'))
model.add(Dropout(0.2, name='dropout1'))
model.add(Dense(512, init='normal', name='dense2'))
model.add(Activation('relu', name='relu2'))
model.add(Dropout(0.2, name='dropout2'))
model.add(Dense(10, init='normal', name='dense3'))
model.add(Activation('softmax', name='softmax1'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
# 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图
cbks = [tb_cb]
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy;', score[1])