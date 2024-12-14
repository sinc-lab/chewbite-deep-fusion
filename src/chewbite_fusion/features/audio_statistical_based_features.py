import numpy as np
from scipy.stats import kurtosis
import librosa

from chewbite_fusion.features.base import BaseFeature


class AudioSignalAverage(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.average(w[0])])
        return windows


class AudioSignalStandardDeviation(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.std(w[0])])
        return windows


class AudioSignalSum(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.sum(w[0])])
        return windows


class AudioSignalMin(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.min(w[0])])
        return windows


class AudioSignalMax(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.max(w[0])])
        return windows


class AudioSignalKurtosis(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([kurtosis(w[0])])
        return windows


class AudioSignalEnergy(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([np.sum(np.abs(w[0]) ** 2)])
        return windows


class AudioSignalZeroCrossing(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([librosa.feature.zero_crossing_rate(w[0],
                                                               frame_length=len(w[0]),
                                                               hop_length=len(w[0]) + 1)[0][0]])
        return windows
