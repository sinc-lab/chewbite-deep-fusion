import os
import copy

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations


class DeepSoundBaseRNN:
    ''' Create a RNN. '''
    def __init__(self,
                 batch_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.ghost_dim = 2
        self.padding = "valid"
        self.training_shape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)

        X_pad = []

        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        y = keras.preprocessing.sequence.pad_sequences(
            y,
            padding='post',
            value=self.padding_class,
            dtype=object)

        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y).astype('float32')

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=X,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')

        X = X[0]
        if self.feature_scaling:
            X = (X + 1.0) * 100

        y_pred = self.model.predict(X).argmax(axis=-1)

        return y_pred

    def predict_proba(self, X):
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')

        X = X[0]

        y_pred = self.model.predict(X)

        return y_pred

    def _get_samples_weights(self, y):
        # Get items counts.
        _, _, counts = np.unique(np.ravel(y),
                                 return_counts=True,
                                 return_index=True,
                                 axis=0)

        # Get max without last element (padding class).
        class_weight = counts[:-1].max() / counts

        # Set padding class weight to zero.
        class_weight[self.padding_class] = 0.0
        class_weight = {cls_num: weight for cls_num, weight in enumerate(class_weight)}
        sample_weight = np.zeros_like(y, dtype=float)

        print(class_weight)

        # Assign weight to every sample depending on class.
        for class_num, weight in class_weight.items():
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepSound(DeepSoundBaseRNN):
    def __init__(self,
                 batch_size=5,
                 input_size=4000,
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance of DeepSound arquitecture.
        '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        layers_config = [(32, 18, 3, activations.relu),
                         (32, 9, 1, activations.relu),
                         (128, 3, 1, activations.relu)]

        # Model definition
        cnn = Sequential()
        cnn.add(layers.Rescaling())

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.padding,
                                      data_format=self.data_format))
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2))

        cnn.add(layers.MaxPooling1D(4))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dropout(rate=0.2))

        ffn = Sequential()

        ffn.add(layers.Dense(256, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(128, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(output_size, activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=(None, input_size, 1), name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128, activation="tanh",
                                                            return_sequences=True, dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())
