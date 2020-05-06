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
timer = mypackage.Timer()

class UNet:

    def __init__(self, X_train, Y_train, loss_func="categorical_crossentropy", saved_mode_name="latest_unet.hdf5", batchnorm=True, dropout=0.5):
        self.history          = None
        self.scale_factor     = 0
        self.saved_mode_name  = saved_mode_name
        self.loss_function    = loss_func
        self.optimizer        = Adadelta(learning_rate=1.0, rho=0.95) # Adam(lr=0.001, decay=1e-06)

        self.X_train = X_train
        self.Y_train = self.__preprocess_y(Y_train) # Changes the data to categorical binary matrix

        self.model = self.__get_model(input_shape=X_train.shape, output_units=self.Y_train.shape[-1], batchnorm=batchnorm, dropout=dropout)

    def __preprocess_y(self, Y):
        return to_categorical(Y)

    def summary(self):
        return self.model.summary()
    
    def predict_only_one(self, X_input, Y_labels):
        i = np.random.choice(len(X_input))
        self.predict(X_input[i:i+1], Y_labels=Y_labels[i:i+1])
        return i

    def predict(self, X_input, Y_labels=None, plot=True, return_heat_map=False):
        # load best weights
        self.model.load_weights(self.saved_mode_name)
#         self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        timer.start()
        y_pred_orig = self.model.predict(X_input)
        timer.stop()
        y_pred_test = np.argmax(y_pred_orig, axis=-1)

        if type(Y_labels) is np.ndarray: # if Not None:
            Y_input = self.__preprocess_y(Y_labels)

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
                
        if return_heat_map:
            y_pred_test = (y_pred_test, y_pred_orig)
        return y_pred_test

    def train(self, batch_size=20, epochs=10, monitor='val_accuracy', mode='max', metrics=['accuracy'], clear_output=False, verbose=1, **kwargs):
#         metrics.append(tf.keras.metrics.MeanIoU(num_classes=3))
        # compiling the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=metrics)

        # checkpoint
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor=monitor, verbose=verbose, save_best_only=True, mode=mode)
        callbacks_list = [checkpoint]

        # Y_input = self.__preprocess_y(Y_train)
        X_input = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        Y_input = tf.convert_to_tensor(self.Y_train, dtype=tf.float32)
        # print(f"X_input {X_input.shape}, Y_input {Y_input.shape}")
        self.history = self.model.fit(x=X_input, y=Y_input, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose, **kwargs)
        
        if clear_output:
            IPython.display.clear_output(wait=True)

    def retrain(self, X_train, Y_train, freeze_up_to=0, batch_size=20, epochs=10, loss=None, verbose=1, **kwargs):
        '''Re-trains the model layers.
                X_train = None        # If None then re-uses initialized training data
                Y_train = None        # If None then re-uses initialized training data
                freeze_upto = 0       # Re-trains on all the layers
        '''
        if loss is None:
            loss = self.loss_function
            
        Y_train = self.__preprocess_y(Y_train)

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
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor=monitor, verbose=verbose, save_best_only=True, mode=mode)
        callbacks_list = [checkpoint]

        # Then recompile the model
        #    and then train the model
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.history = self.model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose, **kwargs)

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

        input_layer = Input((S, S, L))
        
        # Arech @ https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/
        #    https://arxiv.org/abs/1505.00853
        leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=1/5.5)
        activation = leaky_relu # Was 'relu'

        # Code from: https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
        """Function to define the UNET Model"""
        # Contracting Path
        c1 = self.__conv2d_block(input_layer, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, activation=activation)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.__conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, activation=activation)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.__conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.__conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, activation=activation)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.__conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, activation=activation)

        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.__conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, activation=activation)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.__conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.__conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, activation=activation)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.__conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, activation=activation)

        outputs = Conv2D(output_units, (1, 1), activation='softmax')(c9)
        model = Model(inputs=input_layer, outputs=outputs)

        return model
