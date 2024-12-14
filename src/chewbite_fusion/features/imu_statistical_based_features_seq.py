import numpy as np
from scipy.stats import kurtosis

from chewbite_fusion.features.base import BaseFeature


class MovementSignalAccAverage(BaseFeature):
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            for w in file:
                windows.append(np.average(w[1:4], axis=1))
            file_windows.append(windows)

        return file_windows


class MovementSignalAccStandardDeviation(BaseFeature):
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            for w in file:
                windows.append(np.std(w[1:4], axis=1))
            file_windows.append(windows)

        return file_windows


class MovementSignalAccMin(BaseFeature):
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            for w in file:
                windows.append(np.min(w[1:4], axis=1))
            file_windows.append(windows)

        return file_windows


class MovementSignalAccMax(BaseFeature):
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            for w in file:
                windows.append(np.max(w[1:4], axis=1))
            file_windows.append(windows)

        return file_windows


class MovementSignalAccAreaStats(BaseFeature):
    ''' Alvarenga SMA (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            sma = np.sum(np.abs(acc), axis=1)
            for w in sma:
                window = []
                window.append(np.mean(w))
                window.append(np.std(w))
                window.append(np.min(w))
                window.append(np.max(w))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccMagnitudeStats(BaseFeature):
    ''' Alvarenga SVM (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            for w in file:
                window = []
                window.append(np.mean(w[10]))
                window.append(np.std(w[10]))
                window.append(np.min(w[10]))
                window.append(np.max(w[10]))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccVariation(BaseFeature):
    ''' Alvarenga movement variation (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            window_len = acc.shape[2]
            acc_x_var = np.abs(acc[:, 0, 1:window_len] - acc[:, 0, 0:window_len - 1])
            acc_y_var = np.abs(acc[:, 1, 1:window_len] - acc[:, 1, 0:window_len - 1])
            acc_z_var = np.abs(acc[:, 2, 1:window_len] - acc[:, 2, 0:window_len - 1])
            acc_var = acc_x_var + acc_y_var + acc_z_var
            for w in acc_var:
                window = []
                window.append(np.mean(w))
                window.append(np.std(w))
                window.append(np.min(w))
                window.append(np.max(w))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccEnergyStats(BaseFeature):
    ''' Alvarenga energy (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            e = np.sum(np.power(acc, 2), axis=1)
            for w in e:
                window = []
                window.append(np.mean(w))
                window.append(np.std(w))
                window.append(np.min(w))
                window.append(np.max(w))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccEntropyStats(BaseFeature):
    ''' Alvarenga entropy (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            e = np.power(np.sum(acc, axis=1) + 1, 2) * np.log(np.power(np.sum(acc, axis=1), 2) + 1)
            for w in e:
                window = []
                window.append(np.mean(w))
                window.append(np.std(w))
                window.append(np.min(w))
                window.append(np.max(w))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccPitchStats(BaseFeature):
    ''' Alvarenga pitch (accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            vector = -acc[:, 0] / np.sqrt(np.power(acc[:, 1], 2) + np.power(acc[:, 2], 2))

            for w in vector:
                pitch = np.degrees(180 / np.pi * np.arctan(w))
                window = []
                window.append(np.mean(pitch))
                window.append(np.std(pitch))
                window.append(np.min(pitch))
                window.append(np.max(pitch))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccRollStats(BaseFeature):
    ''' Alvarenga roll (from accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])

            for w in acc:
                roll = np.degrees(180 / np.pi * np.arctan2(w[1], w[2]))
                window = []
                window.append(np.mean(roll))
                window.append(np.std(roll))
                window.append(np.min(roll))
                window.append(np.max(roll))
                windows.append(window)
            file_windows.append(windows)

        return file_windows


class MovementSignalAccInclinationStats(BaseFeature):
    ''' Alvarenga inclination (from accelerometer). '''
    def transform(self, X, y=None):
        file_windows = []

        for file in X:
            windows = []
            acc = np.array([x[1:4] for x in file])
            vector = np.sqrt(np.power(acc[:, 0], 2) + np.power(acc[:, 1], 2)) / acc[:, 2]

            for w in vector:
                inclination = np.degrees(180 / np.pi * np.arctan(w))
                window = []
                window.append(np.mean(inclination))
                window.append(np.std(inclination))
                window.append(np.min(inclination))
                window.append(np.max(inclination))
                windows.append(window)
            file_windows.append(windows)

        return file_windows
