import numpy as np
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations

from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.models import settings


def get_samples_weights(y):
    # Get items counts.
    _, _, counts = np.unique(np.ravel(y),
                             return_counts=True,
                             return_index=True,
                             axis=0)

    # Get max excluding last element (padding class).
    class_weight = counts[:-1].max() / counts

    # Set padding class weight to zero.
    class_weight = np.append(class_weight, 0.0)
    class_weight = {cls_num: weight for cls_num, weight in enumerate(class_weight)}
    sample_weight = np.zeros_like(y, dtype=float)

    # Assign weight to every sample depending on class.
    for class_num, weight in class_weight.items():
        sample_weight[y == class_num] = weight

    return sample_weight


def get_data_and_preprocess(window_width, window_overlap, audio_sampling_frequency):
    X, y = main(data_source_names=['jm2004', 'jm2014'],
                window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=audio_sampling_frequency)

    X_train = []
    y_train = []

    for dataset in X.keys():
        for signal in X[dataset].keys():
            for window in range(len(X[dataset][signal])):
                X_train.append(X[dataset][signal][window][0])

    for dataset in y.keys():
        for signal in y[dataset].keys():
            y_train.extend(y[dataset][signal])

    target_encoder = LabelEncoder()

    unique_labels = np.unique(np.hstack(y_train))
    target_encoder.fit(unique_labels)

    y_train_enc = target_encoder.transform(y_train)

    X_train = np.array(X_train)
    X_train = np.expand_dims(X_train, axis=2)

    return X_train, y_train_enc

def deep_sound_base_structure(window_width=0.3,
                              window_overlap=0.5,
                              audio_sampling_frequency=6000):

    X_train, y_train_enc = get_data_and_preprocess(window_width,
                                                   window_overlap,
                                                   audio_sampling_frequency)

    dropout_rate = 0.2
    conv_layers_padding = 'valid'
    data_format = 'channels_last'
    input_size_audio = (None,
                        int(audio_sampling_frequency * window_width),
                        1)
    output_size = 5
    n_epochs = 1400
    batch_size = 50

    sound_cnn = Sequential()
    sound_cnn.add(layers.Rescaling())
    sound_cnn.add(layers.Conv1D(32,
                                kernel_size=18,
                                strides=3,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.Conv1D(32,
                                kernel_size=18,
                                strides=3,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.Dropout(rate=dropout_rate))
    sound_cnn.add(layers.Conv1D(32,
                                kernel_size=9,
                                strides=1,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.Conv1D(32,
                                kernel_size=9,
                                strides=1,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.Dropout(rate=dropout_rate))
    sound_cnn.add(layers.Conv1D(128,
                                kernel_size=3,
                                strides=1,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.Conv1D(128,
                                kernel_size=3,
                                strides=1,
                                activation=activations.relu,
                                padding=conv_layers_padding,
                                data_format=data_format))
    sound_cnn.add(layers.MaxPooling1D(4))
    sound_cnn.add(layers.Flatten())

    sound_cnn.add(layers.Dense(256, activation=activations.relu))
    sound_cnn.add(layers.Dense(128, activation=activations.relu))
    sound_cnn.add(layers.Dense(output_size, activation=activations.softmax))


    sound_cnn.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

    sound_cnn.fit(X_train, y_train_enc,
                  epochs=n_epochs,
                  verbose=1,
                  batch_size=batch_size,
                  validation_split=0.2,
                  shuffle=True,
                  sample_weight=get_samples_weights(y_train_enc))

    # Create new model without final dense layers.
    sound_cnn_trained = Sequential()

    for layer in sound_cnn.layers[:-3]:
        sound_cnn_trained.add(layer)

    sound_cnn_trained.compile(optimizer=Adagrad(),
                              loss='sparse_categorical_crossentropy',
                              weighted_metrics=['accuracy'])

    sound_cnn_trained.build(input_shape=input_size_audio)

    path = settings.models_path
    name = f'{window_width}_{window_overlap}.keras'
    model_dump_file_name = f'{path}/{name}'
    sound_cnn_trained.save(model_dump_file_name)
