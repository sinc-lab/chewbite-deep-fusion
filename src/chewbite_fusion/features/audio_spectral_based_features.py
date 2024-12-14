import librosa

from chewbite_fusion.features.base import BaseFeature


class AudioSpectralCentroid(BaseFeature):
    def transform(self, X, y=None):
        windows = []
        for w in X:
            windows.append([librosa.feature.spectral_centroid(y=w[0],
                                                              sr=self.audio_sampling_frequency,
                                                              n_fft=len(w[0]),
                                                              hop_length=len(w[0]) + 1,
                                                              win_length=len(w[0]))[0][0]])
        return windows
