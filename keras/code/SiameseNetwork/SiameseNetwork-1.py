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

在ClassiFilerNet()函数中，可以看到调用了两次FeatureNetwork()函数，keras.models.Model也被使用的两次，
因此生成的input1和input2是两个完全独立的模型分支，参数是不共享的
'''
# ---------------------函数功能区-------------------------

def FeatureNetwork():
    """
    生成特征提取网络

    这是对MNIST数据调整的网络结构，下面注释掉的部分是原始的Matchnet网络中feature network结构
    """

    inp = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput')

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

    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)

    model = Model(inputs=inp, outputs=models)  # 构造一个model
    return model


def ClassiFilerNet():
    """
    生成度量网络和决策网络

    其实MatchNet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)
    """

    input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
    input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
    for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
        layer.name = layer.name + str("_2")

    inp1 = input1.input
    inp2 = input2.input
    merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models


# ---------------------主调区-------------------------
match_net = ClassiFilerNet()
match_net.summary()  # 打印网络结构
plot_model(match_net, to_file='./match_net-1.png', show_shapes=True)  # 网络结构输出成png图片