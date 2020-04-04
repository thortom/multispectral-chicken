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


## GLOBAL VARIABLES
test_ratio = 0.3
windowSize = 100

class HybridSN:

    def __init__(self, X_train, Y_train, X_test, Y_test, saved_mode_name="HybridSN-best-model.hdf5"):
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
            
        self.X_train, self.Y_train = self.__preprocess(X_train, Y_train) # Changes the data to patches and labels to patch-categorical binary matrix
        self.X_test,  self.Y_test  = self.__preprocess(X_test, Y_test)

        self.model = self.__get_model()
    
    def __preprocess(self, X, Y):
        X = X.astype('float16')
        Y = Y.astype('int8')
        if Y.min() != 0:
            Y -= 1
        
        count, n, m, k = X.shape
        if np.sqrt(count) % 1 != 0: # TODO: Remove the need for this
            raise ValueError(f"Number of training samples needs to be a cubic number. Please reduce the samples down to {np.sqrt(count) // 1}")
        window_length = int(np.sqrt(count))
            
        print(f"count, n, m, k = {count, n, m, k}")
        X = self.__make_one_large_image(X, window_length)
        Y = self.__make_one_large_image(Y, window_length)
        print(f"X.shape = {X.shape}")

        print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
        
        return X, Y
    
    def __make_one_large_image(self, images, window_length):
        full_image = []
        for j in range(window_length):
            one_column = np.concatenate([images[i] for i in range(window_length)])
            if full_image == []:
                full_image = one_column
            else:
                full_image = np.concatenate([full_image, one_column], axis=1)
            
        return full_image
    
    def __get_selectable_pixels(self, n, m, margin):
        selectable_pixels = [(r, c) for r in range(margin, n - margin) for c in range(margin, m - margin)]
        return selectable_pixels
    
    # TODO: Implement over sampling 
    def __data_generator(self, data, label, batch_size=1000, aug=None, in_order=False): # batch_size=2000 was to large to fit in memory with 208 channels
        '''Selects randomly batch_size number of pixels as targes and there corresponding patches'''
        # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        # https://www.tensorflow.org/guide/data#consuming_python_generators
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
        # https://www.tensorflow.org/guide/data#consuming_python_generators
        n, m, k = data.shape
        margin = int((self.windowSize - 1) / 2)
        selectable_pixels = self.__get_selectable_pixels(n, m, margin)
        
        while True:
            X, Y = [], []
            patchesData = np.zeros((batch_size, self.windowSize, self.windowSize, k))
            patchesLabels = np.zeros((batch_size))
            for _ in range(batch_size): # TODO: Select randomly from all possible pixels, batch_size number of pixels with out replacement
                r, c = selectable_pixels[np.random.choice(len(selectable_pixels), replace=False)]
                patch = data[r - margin:r + margin + 1, c - margin:c + margin + 1]
                X.append(patch)
                Y.append(label[r-margin, c-margin])
                
            Y = self.label_binarizer.transform(np.array(Y))
            if self.output_units == 2:
                Y = np.hstack((1 - Y, Y)) # This ensures the same format from the binarizer as with output_units > 2
            X = np.array(X)
            X = X.reshape(*(X.shape), 1)
            
            if aug is not None:
                (X, Y) = next(aug.flow(X, labels, batch_size=batch_size))
            
            print(f"yield (X, Y) {(X.shape, Y.shape)}")
            yield (X, Y)

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
        if count != 1:
            raise ValueError(f"The code only supports one image prediciton at a time. Passed input contains {count} images.")
            
        margin = int((self.windowSize - 1) / 2)
        steps_per_epoch = np.ceil((n-margin)**2 / batch_size)
        X_input, Y_input = self.__preprocess(X_input, Y_input)
        gen_input = self.__data_generator(data=X_input,  label=Y_input,  batch_size=batch_size, in_order=True)

        # load best weights
        self.model.load_weights(self.saved_mode_name)
#         self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        y_pred_test = self.model.predict_generator(gen_input, steps=steps_per_epoch)
        y_pred_test = np.argmax(y_pred_test, axis=-1)

#         classification = classification_report(np.argmax(Y_input, axis=-1).flatten(), y_pred_test.flatten())
#         print(classification)

        print(f"Y_input {Y_input.shape}, y_pred_test {y_pred_test.shape}")
        # Plot results
        plt.figure(figsize=(7,7))
        selected = np.random.choice(len(Y_input))
#         plt.imshow(np.argmax(Y_input, axis=-1)[selected])
        plt.imshow(np.squeeze(Y_input))
        plt.title("True label")

        plt.figure(figsize=(7,7))
#         img = plt.imshow(y_pred_test[selected])
        img = plt.imshow(y_pred_test)
        mypackage.Dataset._Dataset__add_legend_to_image(y_pred_test[selected], img)
        plt.title("Predicted labels")
        plt.show()

    def train(self, batch_size=1000, epochs=2, **kwargs):
        # compiling the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # checkpoint
        checkpoint = ModelCheckpoint(self.saved_mode_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        gen_train = self.__data_generator(data=self.X_train, label=self.Y_train, batch_size=batch_size)
        gen_test  = self.__data_generator(data=self.X_test,  label=self.Y_test,  batch_size=batch_size)
        # See augmentation added to the data in NotMyCode/keras-fit-generator
        
        n, m, k = self.X_train.shape
        steps_per_epoch = np.ceil(n * m / batch_size)     # Worked with BS=1000 and steps=2
        val_steps_per_epoch = np.ceil(self.X_train.shape[0]**2 / batch_size)

        self.history = self.model.fit_generator(gen_train, steps_per_epoch=steps_per_epoch, validation_data=gen_test, validation_steps=val_steps_per_epoch, epochs=epochs, callbacks=callbacks_list, **kwargs)

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
        self.history = self.model.fit_generator(gen, batch_size=batch_size, epochs=epochs, callbacks=None, validation_split=0.1, **kwargs)

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

    # TODO: Change this to the original structure with PCA->30
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

        ## convolutional layers
        conv_layer1 = Conv3D(filters=8, strides=(1, 1, 3), kernel_size=(3, 3, 7), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, strides=(1, 1, 3), kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
        conv_layer3 = Conv3D(filters=32, strides=(1, 1, 3), kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
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
