import numpy as np

from chewbite_fusion.features.base import BaseFeature


class CBIAFeatures(BaseFeature):
    def transform(self, X, y=None):
        windows = []

        for window in X:
            audio_window = window[0]
            max_signal = np.max(audio_window)

            signal_duration = np.zeros_like(audio_window)
            signal_duration[:-1] = (audio_window[:-1] >= (0.15 * max_signal)) + 0

            diff_sign = np.sign(np.diff(audio_window))
            diff_sign = np.append(diff_sign, [0])

            signal_diff = signal_duration * diff_sign

            # Shape index
            shape_index = np.count_nonzero(np.abs(np.diff(signal_diff)) > 1)

            # Duration
            duration = np.sum(signal_duration)

            # Amplitude
            amplitude = np.max(np.abs(audio_window))

            # Symmetry (Adapted in order to capture non-centered events.)
            max_index = np.argmax(audio_window)
            symmetry = np.trapz(audio_window[:max_index]) / np.trapz(audio_window)

            windows.append([shape_index,
                            duration,
                            amplitude,
                            symmetry])

        return windows
