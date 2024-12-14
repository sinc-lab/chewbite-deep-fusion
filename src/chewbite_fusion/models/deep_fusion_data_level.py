import os
import copy

import numpy as np
from scipy import interpolate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations


class DeepFusionDataLevelBase:
    ''' Create a data level fusion network. '''
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
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling

    def _preprocess(self, X, y=None):
        X_pad = []

        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        if y is not None:
            y = keras.preprocessing.sequence.pad_sequences(
                y,
                padding='post',
                value=self.padding_class,
                dtype=object)

        X_interpolated = []
        for channel in range(len(X_pad)):
            if channel == 0:
                X_interpolated.append(X_pad[channel])
            else:
                channel_files = []
                for file in range(len(X_pad[channel])):
                    file_windows = []
                    for window in range(len(X_pad[channel][file])):
                        window_values = X_pad[channel][file][window]
                        interpolator = interpolate.interp1d(
                            list(range(0, len(window_values))),
                            window_values,
                            kind='nearest')
                        interpolated_window = interpolator(
                            np.linspace(0, len(window_values) - 1, len(X_pad[0][0][0])))
                        file_windows.append(interpolated_window)
                    channel_files.append(file_windows)
                X_interpolated.append(channel_files)

        X = np.asarray(X_interpolated).astype('float32')

        if y is not None:
            y = np.asarray(y).astype('float32')

        return X, y

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X, y = self._preprocess(X, y)

        X = np.moveaxis(X, 0, -1)

        if self.training_reshape:
            sequences_length = 46
            new_shape = (int(X.shape[0] * X.shape[1] // sequences_length),
                         sequences_length,
                         X.shape[2],
                         X.shape[3])
            X = np.resize(X, new_shape)
            y = np.resize(y, new_shape[:2])

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
        X, _ = self._preprocess(X)

        X = np.moveaxis(X, 0, -1)

        y_pred = self.model.predict(X).argmax(axis=-1)

        return y_pred

    def predict_proba(self, X):
        return self.model.predict(X,
                                  batch_size=self.batch_size)

    def _get_samples_weights(self, y):
        # Get items counts.
        unique_values, counts = np.unique(np.ravel(y),
                                          return_counts=True)

        class_weight = {value: count
                        for value, count
                        in zip(unique_values, counts)
                        if value != self.padding_class}

        # Set weights based on counts (small number of samples, more weight).
        max_count = np.max(list(class_weight.values()))
        class_weight = {k: max_count / v for k, v in class_weight.items()}

        # Set padding class weight to zero.
        class_weight[self.padding_class] = 0

        # Assign weight to every sample depending on class.
        sample_weight = np.zeros_like(y, dtype=float)

        for class_num, weight in class_weight.items():
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepFusionDataLevel_m1(DeepFusionDataLevelBase):
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             4000,
                             10),
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
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
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.conv_layers_padding,
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
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128,
                                                            activation="tanh",
                                                            return_sequences=True,
                                                            dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionDataLevel_m2(DeepFusionDataLevelBase):
    ''' Reduce dropout. '''
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             4000,
                             10),
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
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
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.conv_layers_padding,
                                      data_format=self.data_format))

        cnn.add(layers.MaxPooling1D(4))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dropout(rate=0.1))

        ffn = Sequential()

        ffn.add(layers.Dense(256, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.1))
        ffn.add(layers.Dense(128, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.1))
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128,
                                                            activation="tanh",
                                                            return_sequences=True,
                                                            dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionDataLevel_m3(DeepFusionDataLevelBase):
    ''' More layers. '''
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             4000,
                             10),
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        layers_config = [(32, 36, 3, activations.relu),
                         (32, 18, 1, activations.relu),
                         (32, 9, 1, activations.relu),
                         (128, 3, 1, activations.relu)]

        # Model definition
        cnn = Sequential()
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.conv_layers_padding,
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
        ffn.add(layers.Dense(64, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128,
                                                            activation="tanh",
                                                            return_sequences=True,
                                                            dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionDataLevel_m4(DeepFusionDataLevelBase):
    ''' LSTM instead of GRU. '''
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             4000,
                             10),
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
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
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.conv_layers_padding,
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
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.LSTM(128,
                                                             activation="tanh",
                                                             return_sequences=True,
                                                             dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionDataLevel_m5(DeepFusionDataLevelBase):
    ''' Add more layers, padding 'same' and different pooling size. '''
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             4000,
                             10),
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        layers_config = [(8, 18, 1, activations.relu),
                         (16, 9, 1, activations.relu),
                         (32, 3, 1, activations.relu),
                         (64, 3, 1, activations.relu),
                         (128, 3, 1, activations.relu)]

        # Model definition
        cnn = Sequential()
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding='same',
                                      data_format=self.data_format))
            cnn.add(layers.MaxPooling1D(2))
            if ix_l == (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2))

        cnn.add(layers.Flatten())
        cnn.add(layers.Dropout(rate=0.2))

        ffn = Sequential()

        ffn.add(layers.Dense(512, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(256,
                                                            activation="tanh",
                                                            return_sequences=True,
                                                            dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionDataLevel_m6(DeepFusionDataLevelBase):
    def __init__(self,
                 batch_size=5,
                 input_size=(None,
                             3000,
                             10),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        layers_config = [(16, 90, 3, activations.relu),
                         (32, 9, 1, activations.relu),
                         (64, 3, 1, activations.relu)]

        # Model definition
        cnn = Sequential()
        cnn.add(layers.BatchNormalization(axis=-1))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.conv_layers_padding,
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
        ffn.add(layers.Dense(output_size + 1,
                             activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=input_size,
                                              name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128,
                                                            activation="tanh",
                                                            return_sequences=True,
                                                            dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())
