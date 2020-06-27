import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Activation,Input,Dropout,Reshape,MaxPooling2D,UpSampling2D,Conv2DTranspose,Add,AveragePooling2D,Concatenate,SpatialDropout2D
from keras.layers.normalization import BatchNormalization 
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.utils import *
from keras import regularizers
from keras import backend as K
import numpy as np
from keras.models import load_model
import pickle


def dense_block(x): 
    # x = 240,240,64
    
    conv_1 = BatchNormalization()(x)
    conv_1 = Activation("relu")(conv_1)
    feature_map_1 = Conv2D(64, (5,5), padding='same', kernel_regularizer=regularizers.l2(1e-5))(conv_1) #240,240,64

    concat_1 = Concatenate()([feature_map_1, x]) #240,240,128

    conv_2 = BatchNormalization()(concat_1)
    conv_2 = Activation("relu")(conv_2)
    feature_map_2 = Conv2D(64, (5,5), padding='same', kernel_regularizer=regularizers.l2(1e-5))(conv_2) #240,240,64

    concat_2 = Concatenate()([feature_map_2, feature_map_1, x]) #240,240,192

    conv_3 = BatchNormalization()(concat_2)
    conv_3 = Activation("relu")(conv_3)
    output_feature_map = Conv2D(64, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(conv_3) #240,240,64

    return output_feature_map


def quicknat(input_shape, num_classes, pool_size=(2, 2)):
    # input_shape = (240,240,4)
    inputs = Input(shape=input_shape)
    # Encoder layers
    input_conv = Conv2D(64, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(inputs) #240,240,64

    dense_1 = dense_block(input_conv)
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(dense_1)

    dense_2 = dense_block(pool_1)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(dense_2)

    dense_3 = dense_block(pool_2)
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(dense_3)

    dense_4 = dense_block(pool_3)
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(dense_4)

    # Bottleneck Layer
    bottleneck = Conv2D(64, (5,5), padding='same', kernel_regularizer=regularizers.l2(1e-5))(pool_4)
    bottleneck = BatchNormalization()(bottleneck)

    # Decoder Layer
    up5 = MaxUnpooling2D(pool_size)([bottleneck, mask_4])
    up5 = Concatenate()([up5, dense_4])
    dense_5 = dense_block(up5)

    up6 = MaxUnpooling2D(pool_size)([dense_5, mask_3])
    up6 = Concatenate()([up6, dense_3])
    dense_6 = dense_block(up6)

    up7 = MaxUnpooling2D(pool_size)([dense_6, mask_2])
    up7 = Concatenate()([up7, dense_2])
    dense_7 = dense_block(up7)

    up8 = MaxUnpooling2D(pool_size)([dense_7, mask_1])
    up8 = Concatenate()([up8, dense_1])
    dense_8 = dense_block(up8)

    # Classifier Block
    classifier = Conv2D(num_classes, (1,1), padding='same', activation='softmax')(dense_8)

    model = Model(inputs=inputs, outputs=classifier, name="QuickNAT")

    return model
