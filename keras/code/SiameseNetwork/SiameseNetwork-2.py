from keras.models import Sequential
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model

'''
代码包含两部分：

第一部分定义了两个函数
1、FeatureNetwork()生成特征提取网络
2、ClassiFilerNet()生成决策网络或称度量网络

在ClassiFilerNet()函数中加入了判断是否使用共享参数模型功能，
令reuse=True，便使用的是共享参数的模型

关键地方就在，只使用的一次Model，也就是说只创建了一次模型，
虽然输入了两个输入，但其实使用的是同一个模型，因此权重共享的
'''


# ---------------------函数功能区-------------------------
def FeatureNetwork():
    """
    生成特征提取网络

    这是对MNIST数据调整的网络结构，下面注释掉的部分是原始的Matchnet网络中feature network结构
    """

    inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')

    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)

    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    models = Activation('relu')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    # models = Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid')(models)
    # models = Activation('relu')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)

    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)

    model = Model(inputs=inp, outputs=models)  # 构造一个model
    return model


def ClassiFilerNet(reuse=True):  # add classifier Net
    """生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""

    if reuse:
        # inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
        # models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
        # models = Activation('relu')(models)
        # models = MaxPool2D(pool_size=(3, 3))(models)
        #
        # models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
        # # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
        # models = Activation('relu')(models)
        #
        # models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        # models = Activation('relu')(models)
        #
        # models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        # models = Activation('relu')(models)
        #
        # # models = Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid')(models)
        # # models = Activation('relu')(models)
        # # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
        #
        # models = Flatten()(models)
        # models = Dense(512)(models)
        # models = Activation('relu')(models)
        # model = Model(inputs=inp, outputs=models)

        model = FeatureNetwork()

        inp1 = Input(shape=(28, 28, 1))  # 创建输入1
        inp2 = Input(shape=(28, 28, 1))  # 创建输入2

        model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
        model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
        merge_layers = concatenate([model_1, model_2], axis=1)  # 进行融合，使用的是默认的sum，即简单的相加

    else:
        m1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
        m2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
        for layer in m2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
            layer.name = layer.name + str("_2")
        inp1 = m1.input
        inp2 = m2.input
        merge_layers = concatenate([m1.output, m2.output])        # 进行融合，使用的是默认的sum，即简单的相加

    ######################
    ##  整合两个网络结果
    ######################
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models


# ---------------------主调区-------------------------
match_net = ClassiFilerNet()
match_net.summary()  # 打印网络结构
plot_model(match_net, to_file='./match_net-2.png', show_shapes=True)  # 网络结构输出成png图片