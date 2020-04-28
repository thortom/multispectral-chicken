# Code from: https://github.com/gokriznastic/HybridSN

import mypackage

import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

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

from numba import jit

class HybridSN:

    def __init__(self, X_train, Y_train, X_test, Y_test, saved_mode_name="latest_HybridSN.hdf5"):
        self.history          = None
        self.scale_factor     = 0
        self.saved_mode_name  = saved_mode_name
        self.loss_function    = "categorical_crossentropy"
        self.optimizer        = Adam(lr=0.001, decay=1e-06)
        self.windowSize       = 25
        self.K                = X_train.shape[-1] # 30   # The spectral dimension used (PCA.numb_components)
        self.output_units     = len(np.unique(Y_train))        
        self.label_binarizer  = LabelBinarizer()
        self.label_binarizer.fit(np.arange(self.output_units))
            
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test,  self.Y_test  = X_test, Y_test

        self.model = self.__get_model()
    
    def __get_selectable_pixels(self, n, m, margin):
        selectable_pixels = [(r, c) for r in range(margin, n - margin) for c in range(margin, m - margin)]
        return selectable_pixels
    
    def __data_generator(self, data, label, aug=None, in_order=False): # batch_size=2000 was to large to fit in memory with 208 channels
        '''Selects randomly batch_size number of pixels as targes and there corresponding patches'''
        # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        # https://www.tensorflow.org/guide/data#consuming_python_generators
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
        # https://www.tensorflow.org/guide/data#consuming_python_generators
        img_count, n, m, k = data.shape
        margin = int((self.windowSize - 1) / 2)
        selectable_pixels = self.__get_selectable_pixels(n, m, margin)
        
        image_order = []
        # Loop indefinitely
        while True:
            if image_order == []:
                image_order = list(np.random.choice(img_count, img_count, replace=False))
                # print("New order")
                # print(image_order)
            i = image_order.pop()
            X, Y = [], []

            for r, c in selectable_pixels:
                patch = data[i, r - margin:r + margin + 1, c - margin:c + margin + 1]
                X.append(patch)
                Y.append(label[i, r, c]) # The original code had an error here. It was label[r-margin, c-margin]

            Y = self.label_binarizer.transform(np.array(Y))
            if self.output_units == 2:
                Y = np.hstack((1 - Y, Y)) # This ensures the same format from the binarizer as with output_units > 2
            X = np.array(X)
            X = X.reshape(*(X.shape), 1)

            if aug is not None:
                (X, Y) = next(aug.flow(X, labels, batch_size=batch_size))

            # print(f"X: {X.shape}")

            yield (X, Y)   # The batch_size is fixed to windowSize^2

    def __scale_output(self, data):
        if self.scale_factor > 0:
            s = self.scale_factor
            return data[:, s:-s, s:-s]
        else:
            return data

    def summary(self):
        return self.model.summary()

    def predict(self, X_input, Y_input, batch_size=1000):
        count, n, m, k = X_input.shape
        X_input = X_input.reshape(*(X_input.shape), 1)
#         if count != 1:
#             raise ValueError(f"The code only supports one image prediciton at a time. Passed input contains {count} images.")

        # load best weights
        self.model.load_weights(self.saved_mode_name)
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        y_pred = np.zeros((count, n, m))
        for i in range(count):
            margin = int((self.windowSize - 1) / 2)
            selectable_pixels = self.__get_selectable_pixels(n, m, margin)
            # TODO: For each (r, c) do the prediction and collect the predictions to a reconstructed image
            for r, c in selectable_pixels:
                patch = X_input[i:i+1, r - margin:r + margin + 1, c - margin:c + margin + 1]

                prediction = self.model.predict(patch)
                y_pred[i, r, c] = np.argmax(prediction, axis=-1)

        Y_input = Y_input.copy()[:, margin:-margin, margin:-margin]
        y_pred  = y_pred[:, margin:-margin, margin:-margin]
        classification = classification_report(Y_input.flatten(), y_pred.flatten())
        print(classification)

        # Plot results
        plt.figure(figsize=(9, 5))
        plt.subplot(121)
        selected = np.random.choice(len(Y_input))
        plt.imshow(np.squeeze(Y_input[selected]))
        plt.title("True label")

        plt.subplot(122)
        img = plt.imshow(y_pred[selected])
        mypackage.Dataset._Dataset__add_legend_to_image(y_pred[selected], img)
        plt.title("Predicted labels")
        plt.show()
        
        return y_pred

    def train(self, epochs=2, **kwargs):
        # compiling the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # checkpoint
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        gen_train = self.__data_generator(data=self.X_train, label=self.Y_train)
        gen_test  = self.__data_generator(data=self.X_test,  label=self.Y_test)
        # See augmentation added to the data in NotMyCode/keras-fit-generator
        
        count, n, m, k = self.X_train.shape
        steps_per_epoch = count
        val_steps_per_epoch = len(self.X_test)

        self.history = self.model.fit_generator(gen_train, steps_per_epoch=steps_per_epoch, validation_data=gen_test, validation_steps=val_steps_per_epoch, max_queue_size=1, workers=1, epochs=epochs, callbacks=callbacks_list, **kwargs)
        # https://keras.io/models/sequential/#fit_generator

    def retrain(self, X_train, Y_train, freeze_up_to=0, batch_size=20, epochs=10, validation_split=0.1, **kwargs):
        '''Re-trains the model layers.
                X_train = None        # If None then re-uses initialized training data
                Y_train = None        # If None then re-uses initialized training data
                freeze_upto = i       # Re-trains all layers from number i, -i re-trains the last i layers
        '''
        X_input, Y_input = self.__preprocess(self, X, Y)

        # load best weights
        self.model.load_weights(self.saved_mode_name)

        # Do something like this: https://www.tensorflow.org/tutorials/images/transfer_learning
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.model.layers))

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.model.layers[:freeze_up_to]:
            layer.trainable = False
            
        gen = self.__data_generator(data=X_train, label=Y_train)
            
        # Then recompile the model
        #    and then train the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        self.history = self.model.fit_generator(gen, steps_per_epoch=len(X_train), epochs=epochs, callbacks=None, validation_split=0.1, **kwargs)

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

    def __get_model(self):

#         _, windowSize, windowSize, L = input_shape
        S            = self.windowSize
        L            = self.K
        output_units = self.output_units

        ########################################
        ## The model ###########################

        batchnorm = True
        n_filters = 8
        dropout = 0.1

        ## input layer## input layer
        input_layer = Input((S, S, L, 1))
        
        if self.K == 208:
            strides = (1, 1, 3)
        else: # Assumes K == 30 here
            strides = (1, 1, 1)

        ## convolutional layers
        conv_layer1 = Conv3D(filters=8, strides=strides, kernel_size=(3, 3, 7), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, strides=strides, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
        conv_layer3 = Conv3D(filters=32, strides=strides, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
        conv3d_shape = conv_layer3._keras_shape
        conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
        conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

        flatten_layer = Flatten()(conv_layer4)

        ## fully connected layers
        dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

        # define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)

        return model
