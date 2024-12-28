import os
import copy
import logging

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

import augly.audio as audaugs


logger = logging.getLogger('yaer')


class DeepFusionFeatureLevelBase:
    ''' Create a feature level fusion network. '''
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

        X_imu = [i for i in X_pad[1:]]
        X_sound = [X_pad[0]]

        if training:
            sequences_length = 46
            X_imu = np.asarray(X_imu).astype('float32')
            X_sound = np.asarray(X_sound).astype('float32')
            X_imu = np.moveaxis(X_imu, 0, -1)
            X_sound = np.moveaxis(X_sound, 0, -1)

            if self.training_reshape:
                new_shape = (int(X_imu.shape[0] * X_imu.shape[1] // sequences_length),
                             sequences_length,
                             X_imu.shape[2],
                             X_imu.shape[3])
                X_imu = np.resize(X_imu, new_shape)

                new_shape = (int(X_sound.shape[0] * X_sound.shape[1] // sequences_length),
                             sequences_length,
                             X_sound.shape[2],
                             X_sound.shape[3])
                X_sound = np.resize(X_sound, new_shape)

                y = np.resize(y, new_shape[:2])

            y = np.asarray(y).astype('float32')
        else:
            X_imu = np.asarray(X_imu).astype('float32')
            X_imu = np.moveaxis(X_imu, 0, -1)
            X_sound = np.asarray(X_sound).astype('float32')
            X_sound = np.moveaxis(X_sound, 0, -1)

        return X_imu, X_sound, y

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        X_imu, X_sound, y = self._preprocess(X, y, training=True)

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

        self.model.fit(x=[X_sound, X_imu],
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_imu, X_sound, _ = self._preprocess(X)

        y_pred = self.model.predict([X_sound, X_imu]).argmax(axis=-1)

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

        logger.info(class_weight)

        # Assign weight to every sample depending on class.
        sample_weight = np.zeros_like(y, dtype=float)

        for class_num, weight in class_weight.items():
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepFusionFeatureLevel_m1(DeepFusionFeatureLevelBase):
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.MaxPooling1D(2))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
        merge = concatenate([sound_x, imu_x])

        rnn = layers.Bidirectional(layers.GRU(128,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(merge)

        # dense_input = Input(shape=(None, 256))
        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m2(DeepFusionFeatureLevelBase):
    ''' Architecture based on m1, with two dense layers after merge layers. '''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.MaxPooling1D(2))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
        merge = concatenate([sound_x, imu_x])

        dense_intermediate = Dense(256, activation='relu')(merge)
        dense_intermediate = Dense(128, activation='relu')(dense_intermediate)

        rnn = layers.Bidirectional(layers.GRU(128,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(dense_intermediate)

        # dense_input = Input(shape=(None, 256))
        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m3(DeepFusionFeatureLevelBase):
    ''' Architecture proposed based on Yao et al. 2016., which proposes a CNN after merge
        layers.'''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=128,
                                  kernel_size=9,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=128,
                                  kernel_size=2,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
        sound_cnn.add(layers.MaxPooling1D(8))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, imu_x], axis=2)
        merge_reshape_dim = int(merge.get_shape()[-1] / 2)
        merge = layers.Reshape((-1, merge_reshape_dim, 2))(merge)

        fusion_cnn = Sequential()
        fusion_cnn.add(layers.Conv1D(filters=64,
                                     kernel_size=9,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.Conv1D(filters=32,
                                     kernel_size=3,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.MaxPooling1D(8))
        fusion_cnn.add(layers.Flatten())
        fusion_x = layers.TimeDistributed(fusion_cnn)(merge)

        rnn = layers.Bidirectional(layers.GRU(128,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(fusion_x)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m4(DeepFusionFeatureLevelBase):
    ''' Model based on m3, with more features at merge step due to smaller max pooling
        in audio layers. The amount of parameters remains similar to m3 because the
        CNN part after merge has a small number of filters. '''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=256,
                                  kernel_size=9,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=256,
                                  kernel_size=2,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
                                    kernel_size=4,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=4,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, imu_x], axis=2)
        merge_reshape_dim = int(merge.get_shape()[-1] / 2)
        merge = layers.Reshape((-1, merge_reshape_dim, 2))(merge)

        fusion_cnn = Sequential()
        fusion_cnn.add(layers.Conv1D(filters=32,
                                     kernel_size=9,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.Conv1D(filters=16,
                                     kernel_size=3,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.MaxPooling1D(8))
        fusion_cnn.add(layers.Flatten())
        fusion_x = layers.TimeDistributed(fusion_cnn)(merge)

        rnn = layers.Bidirectional(layers.GRU(128,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(fusion_x)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m5(DeepFusionFeatureLevelBase):
    ''' Model based on m4, with more filters/units after merge layers (the total
        amount of parameters is increased by ~10x).'''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=256,
                                  kernel_size=9,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=256,
                                  kernel_size=2,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
                                    kernel_size=4,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=4,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, imu_x], axis=2)
        merge_reshape_dim = int(merge.get_shape()[-1] / 2)
        merge = layers.Reshape((-1, merge_reshape_dim, 2))(merge)

        fusion_cnn = Sequential()
        fusion_cnn.add(layers.Conv1D(filters=64,
                                     kernel_size=9,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.Conv1D(filters=32,
                                     kernel_size=3,
                                     strides=1,
                                     activation=activations.relu,
                                     padding=self.conv_layers_padding,
                                     data_format=self.data_format))
        fusion_cnn.add(layers.MaxPooling1D(4))
        fusion_cnn.add(layers.Flatten())
        fusion_x = layers.TimeDistributed(fusion_cnn)(merge)

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(fusion_x)

        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m6(DeepFusionFeatureLevelBase):
    ''' Model based on m1, with more filters/units after merge layers (the total
        amount of parameters is increased by ~10x). '''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.MaxPooling1D(2))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
        merge = concatenate([sound_x, imu_x])

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

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())


class DeepFusionFeatureLevel_m7(DeepFusionFeatureLevelBase):
    ''' Model based on m6, with different head for acc and gyr data. '''
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 30,
                                 3),
                 input_size_gyr=(None,
                                 30,
                                 3),
                 input_size_mag=None,
                 input_size_audio=(None,
                                   1800,
                                   1),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 dropout_rate=0.0,
                 merging_operation=None):
        ''' Create network instance. '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr
        self.input_size_mag = input_size_mag
        self.merging_operation = merging_operation

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

        # Mag model
        if self.input_size_mag is not None:
            mag_input = Input(shape=input_size_mag)

            mag_cnn = Sequential()
            mag_cnn.add(layers.BatchNormalization(axis=-1))
            mag_cnn.add(layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
            mag_cnn.add(layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
            mag_cnn.add(layers.Dropout(rate=dropout_rate))
            mag_cnn.add(layers.MaxPooling1D(2))
            mag_cnn.add(layers.Flatten())

            mag_x = layers.TimeDistributed(mag_cnn)(mag_input)
        else:
            mag_x = None
            mag_input = None

        # Sound model
        deep_sound_input = Input(shape=input_size_audio)

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
        if self.input_size_mag is not None:
            merge = merging_operation([sound_x, acc_x, gyr_x, mag_x])
        else:
            if self.merging_operation:
                logger.info('Merging using operation', str(self.merging_operation))
                merge_1 = self.merging_operation([acc_x, gyr_x])
                merge = concatenate([sound_x, merge_1])
            else:
                merge = concatenate([sound_x, acc_x, gyr_x])

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

        if self.input_size_mag is not None:
            model = Model(inputs=[deep_sound_input,
                                  acc_input,
                                  gyr_input,
                                  mag_input], outputs=output)
        else:
            model = Model(inputs=[deep_sound_input,
                                  acc_input,
                                  gyr_input], outputs=output)
            
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())

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

        X_acc = [i for i in X_pad[1:self.input_size_acc[2] + 1]]
        gyr_start_ix = self.input_size_acc[2] + 1
        X_gyr = [i for i in X_pad[gyr_start_ix:gyr_start_ix + self.input_size_gyr[2]]]
        X_mag = None
        if self.input_size_mag is not None:
            mag_start_ix = gyr_start_ix + self.input_size_gyr[2]
            X_mag = [i for i in X_pad[mag_start_ix:mag_start_ix + self.input_size_mag[2]]]
        X_sound = [X_pad[0]]

        if training:
            sequences_length = 46
            X_acc = np.asarray(X_acc).astype('float32')
            X_gyr = np.asarray(X_gyr).astype('float32')
            X_sound = np.asarray(X_sound).astype('float32')
            X_acc = np.moveaxis(X_acc, 0, -1)
            X_gyr = np.moveaxis(X_gyr, 0, -1)
            X_sound = np.moveaxis(X_sound, 0, -1)
            if self.input_size_mag is not None:
                X_mag = np.asarray(X_mag).astype('float32')
                X_mag = np.moveaxis(X_mag, 0, -1)

            if self.training_reshape:
                new_shape = (int(X_acc.shape[0] * X_acc.shape[1] // sequences_length),
                             sequences_length,
                             X_acc.shape[2],
                             X_acc.shape[3])
                X_acc = np.resize(X_acc, new_shape)
                X_gyr = np.resize(X_gyr, new_shape)

                if self.input_size_mag is not None:
                    X_mag = np.resize(X_mag, new_shape)

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
            if self.input_size_mag is not None:
                X_mag = np.asarray(X_mag).astype('float32')
                X_mag = np.moveaxis(X_mag, 0, -1)

        if self.input_size_mag is not None:
            return X_acc, X_gyr, X_mag, X_sound, y
        else:
            return X_acc, X_gyr, X_sound, y

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

        if self.input_size_mag is not None:
            X_acc, X_gyr, X_mag, X_sound, y = self._preprocess(X, y, training=True)
            train_data = [X_sound, X_acc, X_gyr, X_mag]
        else:
            X_acc, X_gyr, X_sound, y = self._preprocess(X, y, training=True)
            train_data = [X_sound, X_acc, X_gyr]

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

        self.model.fit(x=train_data,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        if self.input_size_mag is not None:
            X_acc, X_gyr, X_mag, X_sound, _ = self._preprocess(X)
            y_pred = self.model.predict([X_sound, X_acc, X_gyr, X_mag]).argmax(axis=-1)
        else:
            X_acc, X_gyr, X_sound, _ = self._preprocess(X)
            y_pred = self.model.predict([X_sound, X_acc, X_gyr]).argmax(axis=-1)

        return y_pred


class DeepFusionFeatureLevel_m8(DeepFusionFeatureLevelBase):
    ''' Model based on m6, with dropout. '''
    def __init__(self,
                 batch_size=5,
                 input_size_imu=(None,
                                 30,
                                 2),
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
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)
        # IMU model
        imu_input = Input(shape=input_size_imu)

        imu_cnn = Sequential()
        imu_cnn.add(layers.BatchNormalization(axis=-1))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  strides=1,
                                  activation=activations.relu,
                                  padding=self.conv_layers_padding,
                                  data_format=self.data_format))
        imu_cnn.add(layers.Dropout(rate=dropout_rate))
        imu_cnn.add(layers.MaxPooling1D(2))
        imu_cnn.add(layers.Flatten())

        imu_x = layers.TimeDistributed(imu_cnn)(imu_input)

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
        merge = concatenate([sound_x, imu_x])

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

        model = Model(inputs=[deep_sound_input,
                              imu_input], outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())