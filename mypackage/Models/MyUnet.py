#################################################################################
# Semantic Segmentation UNet alternatives:                                      #
# https://heartbeat.fritz.ai/a-2019-guide-to-semantic-segmentation-ca8242f5a7fc #
#################################################################################

# TODO: Look into any of these or checkout the NotMyCode/Unet folder
# https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
# https://github.com/zhixuhao/unet/blob/master/model.py

# Code from: https://github.com/gokriznastic/HybridSN
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral

import mypackage

from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add

# Code from: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
def conv2d_block(input_tensor, n_filters, kernel_size = 3, strides=1, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides=strides,\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
#     # second layer
#     x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides=2,\
#               kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
    return x

def run():
    ytrain = np_utils.to_categorical(ytrain)
    ytrain.shape

    S = windowSize
    L = K
    output_units = ytrain.shape[-1]

    ########################################
    ## The model

    batchnorm = True
    n_filters = 8
    dropout = 0.1

    ## input layer
    input_layer = Input((S, S, L, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(n_filters * 1, kernel_size=(3, 3, 7), strides=(1, 1, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(n_filters * 2, kernel_size=(3, 3, 5), strides=(2, 2, 3), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(n_filters * 4, kernel_size=(3, 3, 3), strides=(2, 2, 3), activation='relu')(conv_layer2)
    conv3d_shape = conv_layer3._keras_shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

    # conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    c4 = conv2d_block(conv_layer3, n_filters * 8, kernel_size = 3, strides=2, batchnorm = batchnorm)
    # p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(c4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, strides=2, batchnorm = batchnorm)
    print(f"c4: {c4._keras_shape}")

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    print(f"u6: {u6._keras_shape}")
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    # u7 = concatenate([u7, conv_layer3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    # u8 = concatenate([u8, conv_layer2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    # u9 = concatenate([u9, conv_layer1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(output_units, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=input_layer, outputs=outputs)
    model.summary()
    
    return model