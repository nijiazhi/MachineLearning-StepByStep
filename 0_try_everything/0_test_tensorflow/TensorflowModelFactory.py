#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 16:53
# @Author  : kejiwang
# @Site    : 
# @File    : TensorflowModelFactory.py
# @Software: PyCharm
import keras
#from tensorflow import keras
from keras.models import Model
from keras.layers import Concatenate, Dropout, Dense, LSTM, Input, Embedding, concatenate, Reshape, BatchNormalization, \
    Conv2D, Flatten
from keras.optimizers import Adagrad
import tensorflow as tf

def acc(y,y_conv):
    correct_prediction = tf.greater(tf.square(y-y_conv),0.1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
def error(y,y_conv):
    correct_prediction = tf.square(y-y_conv)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
prams = [8,8,6]

def dnn_net_keras_():#prams = [8,8,6]
    model = keras.models.Sequential ([
        keras.layers.Dense (prams[0],
                            activation=tf.nn.leaky_relu,input_shape=(11,)),
        keras.layers.Dense (prams[1],
                            activation=tf.nn.leaky_relu),
        keras.layers.Dense (prams[2],
                            activation=tf.nn.leaky_relu),
        keras.layers.Dense (1, activation=tf.nn.leaky_relu)
    ])

    model.compile (optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[ acc])#'accuracy', 'binary_crossentropy'
    model.summary ()
    return model




##USE IN DNN 2019-4-8 15:49:17
def dnn_net_keras_dropoutb(drop_radio=0.5, param=None, input=11):
    if param is None:
        param = prams
    print (param)
    model = keras.models.Sequential ([
        keras.layers.Dense (int (param[ 0 ] / drop_radio),
                            activation=tf.nn.leaky_relu, input_shape=(input,)),
        keras.layers.BatchNormalization (),
        keras.layers.Dropout (drop_radio),

        keras.layers.Dense (int (param[ 1 ] / drop_radio),
                            activation=tf.nn.leaky_relu),
        keras.layers.BatchNormalization (),
        keras.layers.Dropout (drop_radio),
        keras.layers.Dense (int (param[ 2 ] / drop_radio),
                            activation=tf.nn.leaky_relu),
        keras.layers.BatchNormalization (),
        keras.layers.Dropout (drop_radio),
        keras.layers.Dense (1, activation=tf.nn.leaky_relu)
    ])
    model.compile (optimizer='adam',
                   loss='mean_squared_error',
                   metrics=[ acc, error ])  # 'accuracy', 'binary_crossentropy'
    model.summary ()#print summary
    return model

def dnn_net_keras_dropoutb_embedding(drop_radio=0.5, param=None, input=11):
    if param is None:
        param = prams
    print(param)
    input = Input (shape=(input,))
    out = Dense (11, ) (input)
    out = Dense (11, ) (out)
    merge = concatenate ([ input, out ])
    out = Dense (int (param[ 0 ] / drop_radio),
                 activation=tf.nn.leaky_relu, input_shape=(input,)) (merge)
    out = BatchNormalization () (out)
    out = Dropout (drop_radio) (out)
    out = Dense (int (param[ 1 ] / drop_radio),
                 activation=tf.nn.leaky_relu, input_shape=(input,)) (out)
    out = BatchNormalization () (out)
    out = Dropout (drop_radio) (out)
    out = Dense (int (param[ 2 ] / drop_radio),
                 activation=tf.nn.leaky_relu, input_shape=(input,)) (out)
    out = BatchNormalization () (out)
    out = Dropout (drop_radio) (out)
    out = Dense (1,
                   activation=tf.nn.leaky_relu, input_shape=(input,)) (out)
    model = Model (inputs=input, outputs=out)
    model.compile (optimizer='adam',
                   loss='mean_squared_error',
                   metrics=[ acc, error ])  # 'accuracy', 'binary_crossentropy'
    model.summary ()
    return model
def sample_model():
    first_input = Input(shape=(11,))
    first_dense = Dense(11, )(first_input)
    dense1 = Dense(6, )(first_dense)
    dense2 = Dense(1, )(dense1)

    second_input = Input(shape=(11,))
    second_dense = Dense(1, )(second_input)

    merge_one = concatenate([dense2, second_dense])

    third_input = Input(shape=(1,))
    merge_two = concatenate([merge_one, third_input])
    #dense = Dense(1, )(merge_two)
    dense = keras.layers.Dense (1, ) (merge_two)
    model = Model(inputs=[first_input, second_input, third_input], outputs=dense)
    ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=ada_grad, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def alex_net_keras(filter =None, drop_radio=0.5, param=None,size =10,showSummary= False):
    alex_filter =[2,4,16,8]
    if param is None:
        param = prams
    if filter is None:
        filter = alex_filter
    model = keras.models.Sequential ([
        keras.layers.Reshape((size,size,1),input_shape=(size*size,)),
        keras.layers.Conv2D (filter[ 0 ], 3),
        keras.layers.BatchNormalization (),

        keras.layers.Conv2D (filter[ 1 ], 3, padding='same'),
        keras.layers.BatchNormalization (),
        #keras.layers.L
        keras.layers.Flatten (),

        keras.layers.Dense (filter[ 2 ],activation=tf.nn.leaky_relu),
        keras.layers.Dense (filter[ 3 ],
                            activation=tf.nn.leaky_relu),
        keras.layers.Dense (2, activation=tf.nn.leaky_relu)


    ])
    model.compile (optimizer='adam',
                   loss='mae',
                   metrics=[ ])  # 'accuracy', 'binary_crossentropy'
    if showSummary:
        model.summary ()
    return model
def inception_keras(showSummary= False):
    input_img = Input (shape=(10, 10, 1,))

    tower_1 = Conv2D (8, (1, 1), padding='same', activation='relu') (input_img)
    tower_1 = Conv2D (8, (3, 3), padding='same', activation='relu') (tower_1)

    tower_2 = Conv2D (8, (1, 1), padding='same', activation='relu') (input_img)
    tower_2 = Conv2D (8, (5, 5), padding='same', activation='relu') (tower_2)

    tower_3 = Conv2D (8,(7, 7), padding='same', activation='relu') (input_img)
    tower_3 = Conv2D (8, (1, 1), padding='same', activation='relu') (tower_3)

    output = concatenate ([ tower_1, tower_2, tower_3 ])
    output = Dense (8,activation=tf.nn.leaky_relu)(output)
    output = Dense (2,
                                 activation=tf.nn.leaky_relu) (output)
    model = Model (inputs=input_img, outputs=output)
    model.compile (optimizer='adam',
                   loss='mse',
                   metrics=[ 'accuracy' ])
    if showSummary:
        model.summary ()
    return model
defalut_filter = [ 1, 3, 3, 9, 9 ]
def cnn_net_keras(filter =None, drop_radio=0.5, param=None,showSummary= False):
    if param is None:
        param = prams
    if filter is None:
        filter = defalut_filter
    model = keras.models.Sequential ([
        keras.layers.Reshape((10,10,1),input_shape=(100,)),
        keras.layers.Conv2D (filter[ 0 ], (1, 1)),
        keras.layers.BatchNormalization (),

        keras.layers.Conv2D (filter[ 1 ], 5, padding='same'),
        keras.layers.BatchNormalization (),

        keras.layers.Conv2D (filter[ 2 ], (1, 1)),
        keras.layers.BatchNormalization (),

        keras.layers.Conv2D (filter[ 3 ], 3, padding='same'),
        keras.layers.BatchNormalization (),

        keras.layers.Conv2D (filter[ 4 ], (1, 1)),
        keras.layers.BatchNormalization (),

        keras.layers.Flatten (),
        keras.layers.Dense (int (param[ 0 ] / drop_radio),
                            activation=tf.nn.leaky_relu),
        keras.layers.BatchNormalization (),
        keras.layers.Dropout (drop_radio),

        keras.layers.Dense (int (param[ 1 ] / drop_radio),
                            activation=tf.nn.leaky_relu),
        keras.layers.BatchNormalization (),
        keras.layers.Dropout (drop_radio),

        keras.layers.Dense (2, activation=tf.nn.leaky_relu)
    ])
    model.compile (optimizer='adam',
                   loss='mean_squared_error',
                   metrics=[ ])  # 'accuracy', 'binary_crossentropy'
    if showSummary:
        model.summary ()
    return model

def slice(x, start, end):
    """ Define a tensor slice function
    """
    return x[:,start:end]
    #slice_1 = Lambda(slice, arguments={'start': n, 'end': n+11})(sliced)


def share_layer_model(pairwise_input_dim=11, cell_input_dim=100):
    # 定义pairwise模型。 这里在模型外进行切片， 也可以使用模型内切片的方式。
    pairwise_input = Input(shape=(pairwise_input_dim,))
    x = Dense(6, activation=tf.nn.leaky_relu)(pairwise_input)
    x = Dense(6, activation=tf.nn.leaky_relu)(x)
    x = Dense(4, activation=tf.nn.leaky_relu)(x)
    x = Dense(1)(x)
    # x = MaxPooling2D((2, 2))(x)
    # out = Flatten()(x)

    pairwise_model = Model(pairwise_input, x,name='pairwise_model_out')

    # pairwise模型将被共享，包括权重和其他所有
    pairwise_models = []
    pairwise_models_input = []

    for i in range(cell_input_dim):
        pairwise_modeli = Input(shape=(pairwise_input_dim,))
        pairwise_modeli_out = pairwise_model(pairwise_modeli)
        pairwise_models_input.append(pairwise_modeli)
        pairwise_models.append(pairwise_modeli_out)

    #
    print("pairs:", len(pairwise_models))

    concatenated = concatenate(pairwise_models,name = 'concatenate_pairwise_output')
    outputpairwise = concatenated
    cnn = Reshape((10, 10, 1), input_shape=(cell_input_dim,))(concatenated)
    cnn = Conv2D(16, 3, padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(32, 3, strides=2, padding='same')(cnn)  # also can use maxpooling
    cnn = BatchNormalization()(cnn)

    out = Flatten()(cnn)
    out = Dense(24, activation=tf.nn.leaky_relu)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.3)(out)
    out = Dense(2, activation=tf.nn.leaky_relu,name='location_model_out')(out)

    loss_weight =[0.05,1]
    location_model = Model(pairwise_models_input, [outputpairwise,out])
    location_model.compile(optimizer='adam', loss={'location_model_out': 'mse', 'concatenate_pairwise_output': 'categorical_crossentropy'},loss_weights=loss_weight,
                           metrics=['accuracy'])#
    location_model.summary()

    #model.fit([inputdata], [labels_pairwise, labels_location],
    #          epochs=50, batch_size=32)
show = True
"""
dnn_net_keras_()
dnn_net_keras_dropoutb()
dnn_net_keras_dropoutb_embedding()

cnn_net_keras(showSummary=show)
alex_net_keras(showSummary=show)
inception_keras(showSummary=show)
"""
share_layer_model()
