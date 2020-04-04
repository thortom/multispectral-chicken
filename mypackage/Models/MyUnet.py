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

class UNet:

    # TODO: Test out loss_func="dice_loss"
    def __init__(self, X_train, Y_train, loss_func="categorical_crossentropy", saved_mode_name="best-model.hdf5"):
        self.history          = None
        self.scale_factor     = 0
        self.saved_mode_name  = saved_mode_name
        self.loss_function    = loss_func
        self.optimizer        = Adam(lr=0.001, decay=1e-06)

        self.X_train = X_train
        self.Y_train = self.__preprocess_y(Y_train) # Changes the data to categorical binary matrix

        self.model = self.__get_model(input_shape=X_train.shape, output_units=self.Y_train.shape[-1])

    def __preprocess_y(self, Y):
        if Y.min() != 0:
            Y -= 1
        return np_utils.to_categorical(Y)

    def __scale_input(self, data, add_dim=False):
        if self.scale_factor > 0:
            s = self.scale_factor
            data = np.pad(data, ((0, 0), (s, s), (s, s), (0, 0)), mode='constant', constant_values=0)
        elif self.scale_factor < 0:
            s = -self.scale_factor
            data = data[:, s:-s, s:-s]
        
        if add_dim:
            data = data.reshape(*(data.shape), 1)
        return data

    def __scale_output(self, data):
        if self.scale_factor > 0:
            s = self.scale_factor
            return data[:, s:-s, s:-s]
        else:
            return data

    def summary(self):
        return self.model.summary()

    def predict(self, X_input, Y_labels=None):
        X_input = self.__scale_input(X_input, add_dim=True)

        # load best weights
        self.model.load_weights(self.saved_mode_name)
#         self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        y_pred_test = self.model.predict(X_input)
        y_pred_test = np.argmax(y_pred_test, axis=-1)

        if type(Y_labels) is np.ndarray: # if Not None:
            Y_input = self.__scale_input(self.__preprocess_y(Y_labels))

            # target_names = ['Belt','Meat','Plastic']
            classification = classification_report(np.argmax(Y_input, axis=-1).flatten(), y_pred_test.flatten())
            print(classification)

            # Plot results
            plt.figure(figsize=(7,7))
            selected = np.random.choice(len(Y_input))
            plt.imshow(np.argmax(Y_input, axis=-1)[selected])
            plt.title("True label")

            plt.figure(figsize=(7,7))
            img = plt.imshow(y_pred_test[selected])
            mypackage.Dataset._Dataset__add_legend_to_image(y_pred_test[selected], img)
            plt.title("Predicted labels")
            plt.show()

    def train(self, batch_size=20, epochs=10, **kwargs):
        # compiling the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # checkpoint
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        X_input = self.__scale_input(self.X_train, add_dim=True)
        Y_input = self.__scale_input(self.Y_train)
        self.history = self.model.fit(x=X_input, y=Y_input, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, **kwargs)

    def retrain(self, X_train, Y_train, freeze_up_to=0, batch_size=20, epochs=10, **kwargs):
        '''Re-trains the model layers.
                X_train = None        # If None then re-uses initialized training data
                Y_train = None        # If None then re-uses initialized training data
                freeze_upto = 0       # Re-trains on all the layers
        '''
        X_input = self.__scale_input(X_train, add_dim=True)
        Y_input = self.__scale_input(self.__preprocess_y(Y_train))

        # load best weights
        self.model.load_weights(self.saved_mode_name)

        # Do something like this: https://www.tensorflow.org/tutorials/images/transfer_learning
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.model.layers))

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.model.layers[:freeze_up_to]:
            layer.trainable = False
            
        # Then recompile the model
        #    and then train the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        self.history = self.model.fit(x=X_input, y=Y_input, batch_size=batch_size, epochs=epochs, callbacks=None, **kwargs)

    def plot_training_results(self):
        # TODO: Do a side by side subplot of these two
        plt.figure(figsize=(7, 7))
        plt.grid()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Training','Validation'], loc='upper right')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.ylim(0, 1.1)
        plt.grid()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Training','Validation'])
        plt.show()

    # Code from: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
    def __conv2d_block(self, input_tensor, n_filters, kernel_size = 3, strides=1, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides=strides,\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides=strides,\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    # TODO: Chagne the model so that the spatial dimension holds the same undtil the spectral has dropped to 8, this allows for concatenation with the upsampling
    def __get_model(self, input_shape, output_units):

        _, windowSize, windowSize, L = input_shape
        S = 64
        if windowSize != S:
            self.scale_factor = (S - windowSize) // 2 # TODO: Fix to handle oddnumbers

        ########################################
        ## The model ###########################

        batchnorm = True
        n_filters = 8
        dropout = 0.1

        ## input layer
        input_layer = Input((S, S, L, 1))

        ## convolutional layers
        conv_layer1 = Conv3D(n_filters * 1, kernel_size=(3, 3, 7), strides=(1, 1, 3), activation='relu', padding='same')(input_layer)
        conv_layer2 = Conv3D(n_filters * 2, kernel_size=(3, 3, 5), strides=(2, 2, 3), activation='relu', padding='same')(conv_layer1)
        conv_layer3 = Conv3D(n_filters * 4, kernel_size=(3, 3, 3), strides=(2, 2, 3), activation='relu', padding='same')(conv_layer2)
        conv3d_shape = conv_layer3._keras_shape
        conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

        # conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

        c4 = self.__conv2d_block(conv_layer3, n_filters * 8, kernel_size = 3, strides=2, batchnorm = batchnorm)
        # p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(c4)

        c5 = self.__conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, strides=2, batchnorm = batchnorm)
        # print(f"c4: {c4._keras_shape}")

        # TODO: Use pix2pix the same as I use in my standard_unet.py
        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.__conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        # u7 = concatenate([u7, conv_layer3]) # TODO: Maybe use https://keras.io/layers/pooling/ # MaxPooling3D to reduce the spectral dimension
        u7 = Dropout(dropout)(u7)
        c7 = self.__conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        # u8 = concatenate([u8, conv_layer2])
        u8 = Dropout(dropout)(u8)
        c8 = self.__conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        # u9 = concatenate([u9, conv_layer1])
        u9 = Dropout(dropout)(u9)
        c9 = self.__conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

        outputs = Conv2D(output_units, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=input_layer, outputs=outputs)
        # model.summary()

        return model
