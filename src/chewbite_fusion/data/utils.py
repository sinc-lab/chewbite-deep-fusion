import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def windows2events(y_pred,
                   window_width=0.5,
                   window_overlap=0.5):
    """ Convert predictions from window-level to event-level.

    Parameters
    ----------
    y_true : tensor or numpy.array[str]
        1D data structure with labels (window-level) for a refence input segment.
    y_pred : tensor or numpy.array[str]
        1D data structure with predictions (window-level) for a refence input segment.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).
    no_event_class : str
        Identifier used to represent the absence of an event of interest.

    Returns
    -------
    df_pred : pandas DataFrame instance.
        DataFrame with start, end and label columns.
    """

    window_starts = np.array(list(range(len(y_pred)))) *\
        (window_width - (window_width * window_overlap))
    window_ends = window_starts + window_width

    df_pred = pd.DataFrame({
        "start": window_starts,
        "end": window_ends,
        "label": y_pred
    })

    df_pred = merge_contiguous(df_pred)
    return df_pred


def merge_contiguous(df):
    """ Given a pandas DataFrame with start, end and label columns it will merge
        contiguous equally labeled. """

    for i in df.index[:-1]:
        next_label = df.loc[i + 1].label
        if next_label == df.loc[i].label:
            df.loc[i + 1, "start"] = df.loc[i].start
            df.drop(i, inplace=True)

    return df


def load_imu_data_from_file(filename):
    ''' Read IMU data stored from Android app and return a Pandas DataFrame instance.

    Params
    ------
    filename : str
        Complete path to the file to be loaded.

    Return
    ------
    df : Data-Frame instance.
        Pandas Data-Frame instance with the following 4 columns:
        - timestamp_sec: timestamp in seconds.
        - {a, g, m}x: signal on x axis.
        - {a, g, m}y: signal on y axis.
        - {a, g, m}z: signal on z axis.
    '''

    # Extract first letter from file name.
    char = os.path.basename(filename)[0]  # a: accelerometer, g: gyroscope, m:magnetometer

    dt = np.dtype([('timestamp', '>i8'),
                   (char + 'x', '>f4'),
                   (char + 'y', '>f4'),
                   (char + 'z', '>f4')])

    with open(filename, 'rb') as f:
        file_data = np.fromfile(f, dtype=dt).byteswap().newbyteorder()
        df = pd.DataFrame(file_data, columns=file_data.dtype.names)

    df["timestamp"] = df["timestamp"] / 1e9
    df.rename(columns={"timestamp": "timestamp_sec"}, inplace=True)
    df["timestamp_relative"] = df.timestamp_sec - df.timestamp_sec.values[0]

    return df


def resample_imu_signal(data, signal_duration_sec, frequency, interpolation_kind='linear'):
    ''' Resample a given signal.

    Params
    ------
    data : Data-Frame instance.
        Data loaded using load_data_from_file method.

    signal_duration_sec : float.
        Total desired duration in seconds (used in order to short resulting signal).

    frequency : int.
        Target frequency.

    interpolation_kind : str.
        Interpolation method used.

    Return
    ------
    df : Data-Frame instance.
        Pandas Data-Frame instance with interpolated signals.
    '''
    axis_cols = [c for c in data.columns if 'timestamp' not in c]

    sequence_end = int(data.timestamp_relative.max()) + 1
    x_values = np.linspace(0, sequence_end,
                           sequence_end * frequency,
                           endpoint=False)

    df = pd.DataFrame({'timestamp_relative': x_values})
    df = df[df.timestamp_relative <= signal_duration_sec]

    for col in axis_cols:
        interpolator = interp1d(data['timestamp_relative'],
                                data[col],
                                kind=interpolation_kind)

        df[col] = interpolator(df.timestamp_relative.values)

    return df
