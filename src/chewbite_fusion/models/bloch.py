import copy

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations


class BlochModel:
    ''' Create a CNN based on Bloch et al. 2023 (https://doi.org/10.3390/s23052611). '''
    def __init__(self,
                 batch_size=32,
                 input_size=(50, 9),
                 output_size=5,
                 n_epochs=1400):
        self.classes_ = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.padding = "valid"

        # Model definition
        cnn = Sequential()
        cnn.add(layers.Conv1D(filters=64,
                              kernel_size=3,
                              strides=1,
                              activation=activations.relu,
                              padding=self.padding,
                              data_format=self.data_format))
        cnn.add(layers.Conv1D(filters=64,
                              kernel_size=3,
                              strides=1,
                              activation=activations.relu,
                              padding=self.padding,
                              data_format=self.data_format))
        cnn.add(layers.Dropout(rate=0.2))
        cnn.add(layers.MaxPooling1D(2))
        cnn.add(layers.Flatten())

        fcnn = Sequential()
        fcnn.add(layers.Dense(100, activation=activations.relu))
        fcnn.add(layers.Dense(output_size, activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size, name='input1'),
                            cnn,
                            fcnn])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        X = np.asarray(X).astype('float32')
        y = np.asarray(y).astype('float32')

        X = np.moveaxis(X, 0, -1)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50)
        ]

        self.model.fit(x=X,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       callbacks=model_callbacks)

    def predict(self, X):
        X = np.asarray(X).astype('float32')

        X = np.moveaxis(X, 0, -1)

        y_pred = self.model.predict(X).argmax(axis=-1)

        return y_pred

    def predict_proba(self, X):
        X = np.asarray(X).astype('float32')

        X = np.moveaxis(X, 0, -1)

        y_pred = self.model.predict(X)

        return y_pred

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))
