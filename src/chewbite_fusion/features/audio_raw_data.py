import numpy as np

from chewbite_fusion.features.base import BaseFeature


class AudioRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_audio = []

        for file in X:
            raw_audio.append([list(window[0]) for window in file])

        return raw_audio
