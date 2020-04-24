#################################################################################
# Semantic Segmentation UNet alternatives:                                      #
# https://heartbeat.fritz.ai/a-2019-guide-to-semantic-segmentation-ca8242f5a7fc #
#################################################################################

# TODO: Look into any of these or checkout the NotMyCode/Unet folder
# https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
# https://github.com/zhixuhao/unet/blob/master/model.py

# Code from: https://github.com/gokriznastic/HybridSN
import IPython
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

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

from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add

import mypackage
timer = mypackage.utils.Timer()

class UNet:

    # TODO: Test out loss_func="dice_loss"
    def __init__(self, X_train, Y_train, loss_func="categorical_crossentropy", saved_mode_name="latest_spectral_unet.hdf5", batchnorm=True, dropout=0.1):
        self.history          = None
        self.scale_factor     = 0
        self.saved_mode_name  = saved_mode_name
        self.loss_function    = loss_func
        self.optimizer        = Adadelta(learning_rate=1.0, rho=0.95) # Adam(lr=0.001, decay=1e-06)

        self.X_train = X_train
        self.Y_train = self.__preprocess_y(Y_train) # Changes the data to categorical binary matrix

        self.model = self.__get_model(input_shape=X_train.shape, output_units=self.Y_train.shape[-1], batchnorm=batchnorm, dropout=dropout)

    def __preprocess_y(self, Y):
        if Y.min() != 0:
            Y -= 1
        return to_categorical(Y)

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
    
    def predict_only_one(self, X_input, Y_labels):
        i = np.random.choice(len(X_input))
        self.predict(X_input[i:i+1], Y_labels=Y_labels[i:i+1])
        return i

    def predict(self, X_input, Y_labels=None, plot=True):
        X_input = self.__scale_input(X_input, add_dim=True)

        # load best weights
        self.model.load_weights(self.saved_mode_name)
#         self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        timer.start()
        y_pred_test = self.model.predict(X_input)
        timer.stop()
        y_pred_test = np.argmax(y_pred_test, axis=-1)

        if type(Y_labels) is np.ndarray: # if Not None:
            Y_input = self.__scale_input(self.__preprocess_y(Y_labels))

            # target_names = ['Belt','Meat','Plastic']
            classification = classification_report(np.argmax(Y_input, axis=-1).flatten(), y_pred_test.flatten())
            print(classification)

            if plot:
                # Plot results
                plt.figure(figsize=(9, 5))
                plt.subplot(121)
                selected = np.random.choice(len(Y_input))
                plt.imshow(np.argmax(Y_input, axis=-1)[selected])
                plt.title("True label")

                plt.subplot(122)
                img = plt.imshow(y_pred_test[selected])
                mypackage.Dataset._Dataset__add_legend_to_image(y_pred_test[selected], img)
                plt.title("Predicted labels")
                plt.show()
            
        return y_pred_test

    def train(self, batch_size=20, epochs=10, monitor='val_accuracy', mode='max', metrics=['accuracy'], clear_output=False, verbose=1, **kwargs):
        # compiling the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=metrics)

        # checkpoint
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor=monitor, verbose=verbose, save_best_only=True, mode=mode)
        callbacks_list = [checkpoint]

        X_input = self.__scale_input(self.X_train, add_dim=True)
        Y_input = self.__scale_input(self.Y_train)
        X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
        Y_input = tf.convert_to_tensor(Y_input, dtype=tf.float32)
        # print(f"X_input {X_input.shape}, Y_input {Y_input.shape}")
        self.history = self.model.fit(x=X_input, y=Y_input, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose, **kwargs)
        
        if clear_output:
            IPython.display.clear_output(wait=True)

    def retrain(self, X_train, Y_train, freeze_up_to=0, batch_size=20, epochs=10, loss=None, **kwargs):
        '''Re-trains the model layers.
                X_train = None        # If None then re-uses initialized training data
                Y_train = None        # If None then re-uses initialized training data
                freeze_upto = 0       # Re-trains on all the layers
        '''
        if loss is None:
            loss = self.loss_function
            
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
            
        # checkpoint
        monitor='val_loss'
        mode='min'
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor=monitor, verbose=1, save_best_only=True, mode=mode)
        callbacks_list = [checkpoint]

        # Then recompile the model
        #    and then train the model
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.history = self.model.fit(x=X_input, y=Y_input, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, **kwargs)

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
    def __conv2d_block(self, input_tensor, n_filters, kernel_size = 3, strides=1, batchnorm=True, activation='relu'):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # One convolution layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides=strides,\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

        return x
    
    def __pool_to_3D(self, layer4D, pooling):
        layer3D = AveragePooling3D(pool_size=pooling, padding='same', data_format='channels_first')(layer4D)
        layer3D_shape = layer3D.shape
        layer3D = Reshape((layer3D_shape[1], layer3D_shape[2], layer3D_shape[3]*layer3D_shape[4]))(layer3D)
        return layer3D

    # TODO: Test out increasing the dropout to 0.5. Evaluate the result on unseen data (/un-labeled)
    def __get_model(self, input_shape, output_units, batchnorm=True, n_filters=8, dropout=0.1):

        _, windowSize, windowSize, L = input_shape
        if windowSize == 32:
            S = windowSize
            # model = self.__small_model(32, L, output_units)
        else:
            S = 64
            if windowSize != S:
                self.scale_factor = (S - windowSize) // 2 # TODO: Fix to handle oddnumbers
            # model = self.__large_model(S, L, output_units)

        input_layer = Input((S, S, L, 1))
        
        # Arech @ https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/
        #    https://arxiv.org/abs/1505.00853
        leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=1/5.5)
        activation = leaky_relu # Was 'relu'

        ## convolutional layers
        conv_layer1 = Conv3D(n_filters * 1, kernel_size=(3, 3, 7), strides=(1, 1, 3), activation=activation, padding='same')(input_layer)
        if batchnorm:
            conv_layer1 = BatchNormalization()(conv_layer1)
        conv_layer1 = Dropout(dropout)(conv_layer1)
        conv_layer2 = Conv3D(n_filters * 2, kernel_size=(3, 3, 5), strides=(2, 2, 3), activation=activation, padding='same')(conv_layer1)
        if batchnorm:
            conv_layer2 = BatchNormalization()(conv_layer2)
        conv_layer2 = Dropout(dropout)(conv_layer2)
        conv_layer3 = Conv3D(n_filters * 4, kernel_size=(3, 3, 3), strides=(2, 2, 3), activation=activation, padding='same')(conv_layer2)
        if batchnorm:
            conv_layer3 = BatchNormalization()(conv_layer3)
        conv_layer3 = Dropout(dropout)(conv_layer3)
        conv3d_shape = conv_layer3.shape
        r3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

        c4 = self.__conv2d_block(r3, n_filters * 8, kernel_size = 3, strides=2, batchnorm=batchnorm, activation=activation)
        # p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(c4)

        c5 = self.__conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, strides=2, batchnorm=batchnorm, activation=activation)

        # TODO: Use pix2pix the same as I use in my standard_unet.py
        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.__conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm, activation=activation)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        # Concat with the third layer
        c3 = self.__pool_to_3D(conv_layer3, pooling=(1, 2, 4))
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.__conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm, activation=activation)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        # Concat with the second layer
        c2 = self.__pool_to_3D(conv_layer2, pooling=(1, 6, 5))
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.__conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm, activation=activation)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        # Concat with the first layer
        c1 = self.__pool_to_3D(conv_layer1, pooling=(1, 9, 8))
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.__conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm, activation=activation)

        outputs = Conv2D(output_units, (1, 1), activation='softmax')(c9)
        # Changed from sigmoid to softmax based on these: https://stackoverflow.com/questions/57253841/from-logits-true-and-from-logits-false-get-different-training-result-for-tf-loss - https://www.quora.com/Why-is-it-better-to-use-Softmax-function-than-sigmoid-function
        model = Model(inputs=input_layer, outputs=outputs)

        return model
