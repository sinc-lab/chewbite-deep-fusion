# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd
from scipy import signal
import librosa
import more_itertools

from chewbite_fusion.data.cache_manager import DatasetCache
from chewbite_fusion.data import utils_data_sources as utils


logger = logging.getLogger(__name__)


def main(data_source_names=['zavalla2022'],
         audio_sampling_frequency=8000,
         movement_sampling_frequency=100,
         window_width=0.5,
         window_overlap=0.5,
         label_overlapping_threshold=0.5,
         filter_noises=True,
         include_movement_magnitudes=False,
         no_event_class_name='no-event',
         filters=None,
         invalidate_cache=False):
    """ Run data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Parameters
        ----------
        data_source_names : list of str
            List of all data source names to be included in the final dataset.
            At this moment there is one option valid: 'zavalla2022'.
        audio_sampling_frequency : int or float
            Sampling frequency of audio source files.
        movement_sampling_frequency : int or float
            Sampling frequency of IMU source files.
        window_width : float
            Size of window in seconds used to split signals.
        window_overlap : float
            Overlapping proportion between to consecutive windows (0.00 - 1.00).
        label_overlapping_threshold : float
            Minimun threshold to assign a label to frame w.r.t. window width (0.00 - 1.00).
        filter_noises : bool
            Define if parts of original signals which include noises are included.
        include_movement_magnitudes : bool
            Define if magnitudes of IMU data are calculated.
        no_event_class_name : str
            Class name to represent the absense of an event of interest.
        filters : list of tuples.
            List of filters, channels and a flag to indicate if applied to movement signals.
            For example, [(signal.butter(10, 15, 'hp'), [0, 1, 2], True)]
                apply a 15th order high-pass Butterworth filter to acceleromter x, y and z axis.
        invalidate_cache : bool
            Force to update cache.

        Returns
        -------
        X : Dictionary-like object, with data sources as keys.
            Each value represent segments of data, and include all extracted windows.
            Example:
            X = {
                'zavalla2022': {
                    'segment_1_cel_1': [
                        [
                                                    # window 1
                            [0.83, 0.55, 0.21],     # audio mono
                            [0.0, 0.1, 0.17],       # acc x
                            [0.0, 0.52, 0.49],      # acc y
                            [0.0, -0.07, -0.14],    # acc z
                            [0.0, 0.1, 0.17],       # gyr x
                            [0.0, 0.52, 0.49],      # gyr y
                            [0.0, -0.07, -0.14],    # gyr z
                            [0.0, 0.1, 0.17],       # mag x
                            [0.0, 0.52, 0.49],      # mag y
                            [0.0, -0.07, -0.14],    # mag z
                                                    # OPTIONAL
                            [0.2, -0.08, 1.15],     # acc magnitude
                            [0.0, 0.0, 0.1],        # gyr magnitude
                            [1.0, 0.97, 0.89],      # mag magnitude
                        ],
                        [
                            ...
                        ]
                    ]
                }
            }
        y : Dictionary-like object, with data sources as keys.
            Each value represent segments of data, and include labels for each window.
            Example:
            y = {
                'zavalla2022': {
                    'segment_1_cel_1': [
                        'no-event',                 # window 1
                        'no-event',                 # window 2
                        'chew',                     # window 3
                        ...
                    ],
                    ...
                }
            }
    """
    logger = logging.getLogger(__name__)

    cache = DatasetCache()

    # Try to retrieve elements from cache.
    cache_item = cache.load(
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        movement_sampling_frequency=movement_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        include_movement_magnitudes=include_movement_magnitudes,
        no_event_class_name=no_event_class_name,
        filters=filters)

    if cache_item and not invalidate_cache:
        logger.info('*** Retrieving dataset from cache ! ***')
        (X, y) = cache_item

        return X, y

    logger.info('*** Creating dataset from scratch ! ***')
    available_datasets = utils.list_datasets()

    for data_source_name in data_source_names:
        assert data_source_name in available_datasets, \
            f'Provided data source name {data_source_name} not available.'

    assert (audio_sampling_frequency * window_width) % 5 == 0, \
        '''Incompatible audio sampling frequency and window width
           (Validation condition: audio_sampling_frequency * window_width) % 5).'''

    assert (audio_sampling_frequency * window_width * (1 - window_overlap)) % 5 == 0, \
        '''Incompatible audio sampling frequency and window overlap
           (Validation condition:
           audio_sampling_frequency * window_width * (1 - window_overlap)) % 5).'''

    assert (movement_sampling_frequency * window_width) % 5 == 0, \
        '''Incompatible movement sampling frequency and window width
           (Validation condition: movement_sampling_frequency * window_width) % 5).'''

    assert (movement_sampling_frequency * window_width * (1 - window_overlap)) % 5 == 0, \
        '''Incompatible movement sampling frequency and window overlap
           (Validation condition:
           movement_sampling_frequency * window_width * (1 - window_overlap)) % 5).'''

    X = {}
    y = {}

    for dataset in data_source_names:
        segment_files = utils.get_files_in_dataset(available_datasets[dataset])

        X_dataset_segments = {}
        y_dataset_segments = {}
        for segment in segment_files:
            segment_name = os.path.basename(segment[0]).split('.')[0]
            logger.info("> Processing segment: %s", segment_name)

            # Read audio file.
            audio_signal, sf = librosa.load(segment[0])
            audio_signal = librosa.resample(y=audio_signal,
                                            orig_sr=sf,
                                            target_sr=audio_sampling_frequency)

            dataset_has_movement_data = len(segment) > 2

            # Read IMU files.
            imu_data = []

            if dataset_has_movement_data:
                for i in range(1, 10):
                    signal_axis_values = pd.read_csv(segment[i],
                                                    names=['axis_value']).axis_value.values
                    data_sampling_frequency = available_datasets[dataset].imu_sampling_frequency
                    if data_sampling_frequency != movement_sampling_frequency:
                        sampling_relation = data_sampling_frequency / movement_sampling_frequency

                        signal_decimated = signal.decimate(signal_axis_values,
                                                        int(sampling_relation))
                        imu_data.append(signal_decimated)
                    else:
                        imu_data.append(signal_axis_values)

                if include_movement_magnitudes:
                    accelerometer_magnitude = \
                        np.sqrt(imu_data[0] ** 2 + imu_data[1] ** 2 + imu_data[2] ** 2)
                    imu_data.append(accelerometer_magnitude)
                    gyroscope_magnitude = \
                        np.sqrt(imu_data[3] ** 2 + imu_data[4] ** 2 + imu_data[5] ** 2)
                    imu_data.append(gyroscope_magnitude)
                    magnetometer_magnitude = \
                        np.sqrt(imu_data[6] ** 2 + imu_data[7] ** 2 + imu_data[8] ** 2)
                    imu_data.append(magnetometer_magnitude)

            if filters:
                for filter in filters:
                    for channel in filter[1]:
                        filter_method = filter[0]
                        if filter[2] and dataset_has_movement_data:
                            imu_data[channel] = filter_method(imu_data[channel])
                        else:
                            audio_signal = filter_method(audio_signal)

            # Read labels file.
            df_segment_labels = pd.read_csv(
                segment[-1],
                sep='\t',
                names=["start", "end", "jm_event"])

            general_mask = df_segment_labels.jm_event
            df_segment_labels.loc[general_mask == 'u', 'jm_event'] = 'unknown'
            df_segment_labels.loc[general_mask == 'b', 'jm_event'] = 'bite'
            df_segment_labels.loc[general_mask == 'c', 'jm_event'] = 'grazing-chew'
            df_segment_labels.loc[general_mask == 'r', 'jm_event'] = 'rumination-chew'
            df_segment_labels.loc[general_mask == 'x', 'jm_event'] = 'chewbite'

            # Get windows from signals.
            audio_windows = get_windows_from_audio_signal(
                audio_signal,
                sampling_frequency=audio_sampling_frequency,
                window_width=window_width,
                window_overlap=window_overlap)

            imu_windows = []
            if dataset_has_movement_data:
                imu_windows = get_windows_from_imu_signals(
                    imu_data,
                    sampling_frequency=movement_sampling_frequency,
                    window_width=window_width,
                    window_overlap=window_overlap)

                if len(audio_windows) - len(imu_windows) == 1:
                    logger.info('Removing last audio window in order to align with imu windows !')
                    audio_windows = audio_windows[:-1]

                assert len(audio_windows) == len(imu_windows),\
                    f'''Number of windows mismatched between audio
                        ({len(audio_windows)}) and IMU data ({len(imu_windows)}).'''

            # Get window labels.
            window_labes = get_windows_labels(
                df_segment_labels,
                len(audio_windows),
                window_width=window_width,
                window_overlap=window_overlap,
                label_overlapping_threshold=label_overlapping_threshold,
                no_event_class_name=no_event_class_name)

            segment_windows = []
            imu_channels = 0

            if dataset_has_movement_data:
                imu_channels = len(imu_windows[0])

            for i in range(len(audio_windows)):
                window_channels = []
                window_channels.append(audio_windows[i])

                if dataset_has_movement_data:
                    for c_i in range(imu_channels):
                        window_channels.append(imu_windows[i][c_i])
                segment_windows.append(window_channels)

            # Construct final results.
            X_dataset_segments[segment_name] = segment_windows
            y_dataset_segments[segment_name] = window_labes

        X[dataset] = X_dataset_segments
        y[dataset] = y_dataset_segments

    # Create cache item.
    cache.save(
        X,
        y,
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        movement_sampling_frequency=movement_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        include_movement_magnitudes=include_movement_magnitudes,
        no_event_class_name=no_event_class_name,
        filters=filters)

    return X, y


def get_windows_from_audio_signal(
        signal,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window.

    Parameters
    ----------
    signal : NumPy array
        Signal values.
    sampling_frequency : int
        Number of samples per second.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).

    Returns
    -------
    windows : list of lists.
        Extracted windows.
    '''
    windows = librosa.util.frame(signal,
                                 frame_length=int(sampling_frequency * window_width),
                                 hop_length=int((1 - window_overlap) * int(sampling_frequency *
                                                                           window_width)),
                                 axis=0)

    return windows


def get_windows_from_imu_signals(
        imu_data,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window.

    Parameters
    ----------
    signal : NumPy array
        Signal values.
    sampling_frequency : int
        Number of samples per second.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).

    Returns
    -------
    windows : list of lists.
        Extracted windows.
    '''

    hop_length = int((1 - window_overlap) * int(sampling_frequency * window_width))
    frame_length = int(sampling_frequency * window_width)

    signals = []
    for ix in range(len(imu_data)):
        signals.append(
            librosa.util.frame(imu_data[ix],
                               frame_length=frame_length,
                               hop_length=hop_length,
                               axis=0))

    return list(map(list, zip(*signals)))


def get_windows_labels(
        labels,
        n_windows,
        window_width,
        window_overlap,
        label_overlapping_threshold,
        no_event_class_name):
    ''' Extract labels for each window.

    Parameters
    ----------
    labels : pandas DataFrame instance.
        Labels information including start, end and event.
    n_windows : int
        Number of windows.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Percentage of overlapping between to consecutive windows (0-100%).
    label_overlapping_threshold : float
        Minimun threshold to assign a label to frame w.r.t. window width (0-100%).
    no_event_class_name : str
        Class name to represent the absense of an event of interest.

    Returns
    -------
    window_labels : list
        Corresponding label for each window.
    '''
    window_start = 0
    window_end = window_width

    window_labels = []

    labels['not_used'] = True

    for i in range(n_windows):
        labels_matched = labels[(labels.start <= window_end) & (labels.end >= window_start)]

        if len(labels_matched) > 0:
            overlappings = []
            for index, label in labels_matched.iterrows():
                event_duration = label.end - label.start
                overlap_in_seconds = min(label.end, window_end) - max(label.start, window_start)
                overlappings.append((overlap_in_seconds,
                                     label.jm_event,
                                     index,
                                     event_duration))

            # Sort all labels with overlap.
            overlappings.sort(key=lambda tup: tup[0], reverse=True)

            exist_overlap_for_window = False
            for ix_o, overlap in enumerate(overlappings):
                # If the window contains the entire event, asign the label.
                # event:            |     |
                # window:        |            |
                window_contains_the_event = (overlap[0] / overlap[3]) == 1

                # If overlap % compared to window width reachs the threshold, asign the label.
                # event:        | ------|       - (overlap)
                # window:        |------   |
                relative_overlap = (overlap[0] / window_width)
                overlap_reachs_threshold = relative_overlap >= label_overlapping_threshold

                # If any of created conditions is True, then asign the label to the window.
                if (window_contains_the_event or overlap_reachs_threshold):
                    exist_overlap_for_window = True

                    # If overlap is enough, the label of event with more overlap will be used.
                    window_labels.append(overlap[1])

                    # All events with enough overlap are used.
                    labels.loc[overlap[2], 'not_used'] = False

                    break

            if not exist_overlap_for_window:
                window_labels.append(no_event_class_name)
        else:
            window_labels.append(no_event_class_name)

        window_start = window_start + window_width * (1 - window_overlap)
        window_end = window_start + window_width

    not_used_labels = labels[(labels.jm_event != 'u') & (labels.not_used)]
    if len(not_used_labels) > 0:
        logger.info('Some labels have not been used: %s', str(len(not_used_labels)))
    else:
        logger.info('All labels have been used.')

    return window_labels
