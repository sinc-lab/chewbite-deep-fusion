import os
import copy

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merging import concatenate


class AblationBase:
    def _preprocess(self, X, y=None, training=False):
        X_pad = []

        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        if training:
            y = keras.preprocessing.sequence.pad_sequences(
                y,
                padding='post',
                value=self.padding_class,
                dtype=object)

        X_acc = [i for i in X_pad[1:4]]
        X_gyr = [i for i in X_pad[4:7]]
        X_sound = [X_pad[0]]

        if training:
            sequences_length = 46
            X_acc = np.asarray(X_acc).astype('float32')
            X_gyr = np.asarray(X_gyr).astype('float32')
            X_sound = np.asarray(X_sound).astype('float32')
            X_acc = np.moveaxis(X_acc, 0, -1)
            X_gyr = np.moveaxis(X_gyr, 0, -1)
            X_sound = np.moveaxis(X_sound, 0, -1)

            if self.training_reshape:
                new_shape = (int(X_acc.shape[0] * X_acc.shape[1] // sequences_length),
                             sequences_length,
                             X_acc.shape[2],
                             X_acc.shape[3])
                X_acc = np.resize(X_acc, new_shape)
                X_gyr = np.resize(X_gyr, new_shape)

                new_shape = (int(X_sound.shape[0] * X_sound.shape[1] // sequences_length),
                             sequences_length,
                             X_sound.shape[2],
                             X_sound.shape[3])
                X_sound = np.resize(X_sound, new_shape)

                y = np.resize(y, new_shape[:2])

            y = np.asarray(y).astype('float32')
        else:
            X_acc = np.asarray(X_acc).astype('float32')
            X_acc = np.moveaxis(X_acc, 0, -1)
            X_gyr = np.asarray(X_gyr).astype('float32')
            X_gyr = np.moveaxis(X_gyr, 0, -1)
            X_sound = np.asarray(X_sound).astype('float32')
            X_sound = np.moveaxis(X_sound, 0, -1)

        return X_acc, X_gyr, X_sound, y

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


class DeepFusionAblationA(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, only sound head. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 input_size_gyr=(None,
                                 30,
                                 3),
                 input_size_audio=(None,
                                   1800,
                                   1),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        dropout_rate = 0.0
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr

        deep_sound_input = Input(shape=input_size_audio)

        # Sound model
        sound_cnn = Sequential()
        sound_cnn.add(layers.Rescaling())
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)


        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(sound_x)

        # dense_input = Input(shape=(None, 256))
        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=deep_sound_input, outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        _, _, X_sound, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=150),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=X_sound,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        _, _, X_sound, _ = self._preprocess(X)

        y_pred = self.model.predict(X_sound).argmax(axis=-1)

        return y_pred


class DeepFusionAblationB(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, only IMU head. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 input_size_gyr=(None,
                                 30,
                                 3),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 dropout_rate=0):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr

        # Acc head
        acc_input = Input(shape=input_size_acc)

        acc_cnn = Sequential()
        acc_cnn.add(layers.BatchNormalization(axis=-1))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Dropout(rate=dropout_rate))
        acc_cnn.add(layers.MaxPooling1D(2))
        acc_cnn.add(layers.Flatten())

        acc_x = layers.TimeDistributed(acc_cnn)(acc_input)

        # Gyr head
        gyr_input = Input(shape=input_size_gyr)

        gyr_cnn = Sequential()
        gyr_cnn.add(layers.BatchNormalization(axis=-1))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Dropout(rate=dropout_rate))
        gyr_cnn.add(layers.MaxPooling1D(2))
        gyr_cnn.add(layers.Flatten())

        gyr_x = layers.TimeDistributed(gyr_cnn)(gyr_input)

        merge = concatenate([acc_x, gyr_x])

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(merge)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[acc_input,
                              gyr_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X_acc, X_gyr, _, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=150),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=[X_acc, X_gyr],
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_acc, X_gyr, _, _ = self._preprocess(X)

        y_pred = self.model.predict([X_acc, X_gyr]).argmax(axis=-1)

        return y_pred


class DeepFusionAblationB_a(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, only acc head. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        dropout_rate = 0.0
        self.input_size_acc = input_size_acc

        # Acc head
        acc_input = Input(shape=input_size_acc)

        acc_cnn = Sequential()
        acc_cnn.add(layers.BatchNormalization(axis=-1))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Dropout(rate=dropout_rate))
        acc_cnn.add(layers.MaxPooling1D(2))
        acc_cnn.add(layers.Flatten())

        acc_x = layers.TimeDistributed(acc_cnn)(acc_input)

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(acc_x)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=acc_input, outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X_acc, _, _, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=150),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=X_acc,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_acc, _, _, _ = self._preprocess(X)

        y_pred = self.model.predict(X_acc).argmax(axis=-1)

        return y_pred


class DeepFusionAblationB_b(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, only gyr head. '''
    def __init__(self,
                 batch_size=5,
                 input_size_gyr=(None,
                                 30,
                                 3),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        dropout_rate = 0.0
        self.input_size_gyr = input_size_gyr

        # Gyr head
        gyr_input = Input(shape=input_size_gyr)

        gyr_cnn = Sequential()
        gyr_cnn.add(layers.BatchNormalization(axis=-1))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Dropout(rate=dropout_rate))
        gyr_cnn.add(layers.MaxPooling1D(2))
        gyr_cnn.add(layers.Flatten())

        gyr_x = layers.TimeDistributed(gyr_cnn)(gyr_input)

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(gyr_x)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=gyr_input, outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        _, X_gyr, _, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=150),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=X_gyr,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        _, X_gyr, _, _ = self._preprocess(X)

        y_pred = self.model.predict(X_gyr).argmax(axis=-1)

        return y_pred


class DeepFusionAblationC(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, without RNN part. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 input_size_gyr=(None,
                                 30,
                                 3),
                 input_size_audio=(None,
                                   1800,
                                   1),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights

        dropout_rate = 0.0
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr

        # Acc model
        acc_input = Input(shape=input_size_acc)

        acc_cnn = Sequential()
        acc_cnn.add(layers.BatchNormalization(axis=-1))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Dropout(rate=dropout_rate))
        acc_cnn.add(layers.MaxPooling1D(2))
        acc_cnn.add(layers.Flatten())

        acc_x = layers.TimeDistributed(acc_cnn)(acc_input)
        
        # Gyr model
        gyr_input = Input(shape=input_size_gyr)

        gyr_cnn = Sequential()
        gyr_cnn.add(layers.BatchNormalization(axis=-1))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Dropout(rate=dropout_rate))
        gyr_cnn.add(layers.MaxPooling1D(2))
        gyr_cnn.add(layers.Flatten())

        gyr_x = layers.TimeDistributed(gyr_cnn)(gyr_input)

        deep_sound_input = Input(shape=input_size_audio)

        # Sound model
        sound_cnn = Sequential()
        sound_cnn.add(layers.Rescaling())
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, acc_x, gyr_x])

        # dense_input = Input(shape=(None, 256))
        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(merge)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              acc_input,
                              gyr_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X_acc, X_gyr, X_sound, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=[X_sound, X_acc, X_gyr],
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_acc, X_gyr, X_sound, _ = self._preprocess(X)

        y_pred = self.model.predict([X_sound, X_acc, X_gyr]).argmax(axis=-1)

        return y_pred


class DeepFusionAblationD(AblationBase):
    ''' Ablation model based on DeepFusionFeatureLevel_m6, without dense part. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 input_size_gyr=(None,
                                 30,
                                 3),
                 input_size_audio=(None,
                                   1800,
                                   1),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 dropout_rate=0.0):
        ''' Create network instance. '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr

        # Acc model
        acc_input = Input(shape=input_size_acc)

        acc_cnn = Sequential()
        acc_cnn.add(layers.BatchNormalization(axis=-1))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        acc_cnn.add(layers.Dropout(rate=dropout_rate))
        acc_cnn.add(layers.MaxPooling1D(2))
        acc_cnn.add(layers.Flatten())

        acc_x = layers.TimeDistributed(acc_cnn)(acc_input)
        
        # Gyr model
        gyr_input = Input(shape=input_size_gyr)

        gyr_cnn = Sequential()
        gyr_cnn.add(layers.BatchNormalization(axis=-1))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        gyr_cnn.add(layers.Dropout(rate=dropout_rate))
        gyr_cnn.add(layers.MaxPooling1D(2))
        gyr_cnn.add(layers.Flatten())

        gyr_x = layers.TimeDistributed(gyr_cnn)(gyr_input)

        deep_sound_input = Input(shape=input_size_audio)

        # Sound model
        sound_cnn = Sequential()
        sound_cnn.add(layers.Rescaling())
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, acc_x, gyr_x])

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(merge)

        output = Dense(output_size, activation=activations.softmax)(rnn)

        model = Model(inputs=[deep_sound_input,
                              acc_input,
                              gyr_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X_acc, X_gyr, X_sound, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints,
                                      'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=[X_sound, X_acc, X_gyr],
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_acc, X_gyr, X_sound, _ = self._preprocess(X)

        y_pred = self.model.predict([X_sound, X_acc, X_gyr]).argmax(axis=-1)

        return y_pred
