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

from chewbite_fusion.models.deep_fusion_feature_level import DeepFusionFeatureLevelBase

logger = logging.getLogger('yaer')

class DeepFusionFeatureLevelWithTL_m1(DeepFusionFeatureLevelBase):
    def __init__(self,
                 batch_size=5,
                 input_size_acc=(None,
                                 15,
                                 3),
                 input_size_gyr=(None,
                                 15,
                                 3),
                 input_size_audio=(None,
                                   1800,
                                   1),
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 acc_base_model_path=None,
                 sound_base_model_path=None,
                 layers_to_unfreeze_acc=0,
                 layers_to_unfreeze_audio=0):
        ''' Create network instance. '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)
        dropout_rate = 0.0
        self.input_size_acc = input_size_acc
        self.input_size_gyr = input_size_gyr

        # Acc model
        acc_input = Input(shape=input_size_acc)
        acc_cnn = keras.models.load_model(acc_base_model_path)

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

        # Sound model
        deep_sound_input = Input(shape=input_size_audio)
        sound_cnn = keras.models.load_model(sound_base_model_path)

        # Change parameters to layers index.
        if layers_to_unfreeze_acc != 0:
            layers_to_unfreeze_acc = layers_to_unfreeze_acc * -1

        if layers_to_unfreeze_audio != 0:
            layers_to_unfreeze_audio = layers_to_unfreeze_audio * -1

        for layer in acc_cnn.layers[:layers_to_unfreeze_acc]:
            layer.trainable = False

        for layer in sound_cnn.layers[:layers_to_unfreeze_audio]:
            layer.trainable = False

        acc_x = layers.TimeDistributed(acc_cnn)(acc_input)
        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # Merge models
        # merge input models
        merge = concatenate([sound_x, acc_x, gyr_x])

        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(merge)

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

    def fit(self, X, y):
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))

        self.padding_class = len(self.classes_)

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
        X_acc, X_gyr, X_sound, _ = self._preprocess(X)
        y_pred = self.model.predict([X_sound, X_acc, X_gyr]).argmax(axis=-1)

        return y_pred
