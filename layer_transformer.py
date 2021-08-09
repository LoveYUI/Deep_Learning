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

class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # print("WQ.shape", WQ.shape)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (WQ.shape[2] ** 0.5)
        QK = K.softmax(QK)
        # print("QK.shape", QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

def M_Self_Attention(n,output_dim,input):
    for i in range(n):
        if i == 0:
            MSA = Self_Attention(output_dim)(input)
        else:
            MSA = concatenate((MSA,Self_Attention(output_dim)(input)))

    return MSA

def location_encoder(len,wordvec_len):
    y=[]
    for i in range(len):
        yi=[]
        for j in range(wordvec_len):
            if j%2==0:
                yi.append(np.sin(i/(np.power(10000,(j/wordvec_len)))))
            else:
                yi.append(np.cos(i/(np.power(10000,((j-1)/wordvec_len)))))
        y.append(yi)
    return tf.convert_to_tensor(np.array(y),dtype=float)

def embedding(input,output_dim):
    embeddings = Dense(output_dim)(input)
    loc = Lambda(lambda x: x + location_encoder(49, output_dim))(embeddings)
    return loc

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

get_custom_objects().update({'gelu': Activation(gelu)})

def transformer(input,model="base",is_first=False):
    if model!="base" and model!="large" and model!="huge" and model!= "mnist":
        print("model must be base,large or huge")
        model = "base"
    if model == "mnist":
        if is_first:
            input = embedding(input,64)
        norm=LayerNormalization()(input)
        M_S_A = M_Self_Attention(8, 8, norm)
        M_S_A_2 = Dense(64)(M_S_A)
        add_and_norm1 = add([M_S_A_2, input])
        norm2 = LayerNormalization()(add_and_norm1)
        feed_forward = Dense(128, activation="gelu")(norm2)
        feed_forward1 = Dense(64)(feed_forward)
        add_and_norm2 = add([add_and_norm1, feed_forward1])
    if model == "base":
        if is_first:
            input = embedding(input,768)
        norm=LayerNormalization()(input)
        M_S_A = M_Self_Attention(12, 64, norm)
        M_S_A_2 = Dense(768)(M_S_A)
        add_and_norm1 = add([M_S_A_2, input])
        norm2 = LayerNormalization()(add_and_norm1)
        feed_forward = Dense(3072, activation="gelu")(norm2)
        feed_forward1 = Dense(768)(feed_forward)
        add_and_norm2 = add([add_and_norm1, feed_forward1])
    return add_and_norm2

def Vit(input,outputs_dim,model="base"):
    if model!="base" and model!="large" and model!="huge" and model!="mnist":
        print("model must be base,large or huge")
        model = "base"
    if model == "mnist":
        Vit_1 = transformer(input,"mnist",True)
        mlp = Flatten()(Vit_1)
        outputs_mlp = Dense(32, activation="tanh")(mlp)
        outputs = Dense(outputs_dim, activation="softmax")(outputs_mlp)
    if model == "base":
        Vit_1 = transformer(input, "base", True)
        Vit_2 = transformer(Vit_1)
        Vit_3 = transformer(Vit_2)
        Vit_4 = transformer(Vit_3)
        Vit_5 = transformer(Vit_4)
        Vit_6 = transformer(Vit_5)
        Vit_7 = transformer(Vit_6)
        Vit_8 = transformer(Vit_7)
        Vit_9 = transformer(Vit_8)
        Vit_10 = transformer(Vit_9)
        Vit_11 = transformer(Vit_10)
        Vit_12 = transformer(Vit_11)
        Vit_12 = LayerNormalization(Vit_12)
        mlp = Flatten()(Vit_12)
        outputs_mlp = Dense(32, activation="tanh")(mlp)
        outputs = Dense(outputs_dim, activation="softmax")(outputs_mlp)
    return outputs


#Transformers

if __name__=='__main__':
    inputs=Input(shape=(49,16),dtype='float')
    outputs = Vit(inputs,10,"base")
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr = 1e-1),loss="sparse_categorical_crossentropy",metrics="accuracy")
    print(model.summary())

    #load mnist
    for i in range(6):
        if not i:
            X_train=np.load("./mnist_x"+str(i+1)+".npy")
            Y_train=np.load("./mnist_y"+str(i+1)+".npy")
        else:
            X_train=np.concatenate((X_train,np.load("./mnist_x"+str(i+1)+".npy")),axis=0)
            Y_train=np.concatenate((Y_train,np.load("./mnist_y"+str(i+1)+".npy")),axis=0)
    X_test = np.load("./mnist_x9.npy")
    Y_test = np.load("./mnist_y9.npy")
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    #pre


    X_train=X_train.reshape(48000,49,4,4)
    X_train=X_train.reshape(48000,49,16)
    X_train=X_train/255.0
    Y_train=Y_train.reshape(48000,49)
    Y_train=np.sum(Y_train,axis=1)/49
    X_test=X_test.reshape(12000,49,4,4)
    X_test=X_test.reshape(12000,49,16)
    X_test=X_test/255.0
    Y_test=Y_test.reshape(12000,49)
    Y_test=np.sum(Y_test,axis=1)/49
    print(Y_test.shape)
    print(Y_train.shape)
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=128,epochs=10)
    model.save("./model_mnist_vit")


