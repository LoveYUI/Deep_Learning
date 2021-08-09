from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import *
from keras.datasets import mnist
from keras.optimizers import *
from keras.utils import *
import tensorflow as tf
from keras.activations import relu

def resnet_18(input, label_num):
    assert input.shape[1] == 224 and input.shape[2] == 224

    # before res_block
    conv_0 = Conv2D(input_shape=(224, 224, 3),filters=64, kernel_size=7, strides=2, padding='same')(input)
    maxpool_1 = MaxPooling2D(pool_size=(3, 3),strides=(2, 2), padding='same')(conv_0)

    # res_block_1
    conv_1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(maxpool_1)
    conv_1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_1_1)
    block_1 = add([maxpool_1, conv_1_2])

    # res_block_2
    conv_2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_1)
    conv_2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_2_1)
    block_2 = add([block_1, conv_2_2])

    # res_block_3
    conv_3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_2)
    conv_3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_3_1)
    sub_sample_3_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(block_2)
    block_3 = add([sub_sample_3_1, conv_3_2])

    # res_block_4
    conv_4_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_3)
    conv_4_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_4_1)
    block_4 = add([block_3, conv_4_2])

    # res_block_5
    conv_5_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_4)
    conv_5_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_5_1)
    sub_sample_5_1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same')(block_4)
    block_5 = add([sub_sample_5_1, conv_5_2])

    # res_block_6
    conv_6_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_5)
    conv_6_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6_1)
    block_6 = add([block_5, conv_6_2])

    # res_block_7
    conv_7_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_6)
    conv_7_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_7_1)
    sub_sample_7_1 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='same')(block_6)
    block_7 = add([sub_sample_7_1, conv_7_2])

    # res_block_8
    conv_8_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_7)
    conv_8_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_8_1)
    block_8 = add([block_7, conv_8_2])

    # avgpool
    avg_pool = AveragePooling2D(pool_size=(7, 7))(block_8)
    avg_pool = Flatten()(avg_pool)

    # MLP
    MLP_1 = Dense(256, activation=LeakyReLU(0.3))(avg_pool)
    if label_num == 2:
        output = Dense(1, activation="sigmoid")(MLP_1)
    else:
        output = Dense(label_num, activation="softmax")(MLP_1)

    return output

if __name__ == "__main__":
    input = Input(shape=(224, 224, 3))
    output = resnet_18(input, 3)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
