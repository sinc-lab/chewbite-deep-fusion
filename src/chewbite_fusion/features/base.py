class BaseFeatureBuilder():
    def __init__(self,
                 feature,
                 audio_sampling_frequency,
                 movement_sampling_frequency,
                 preprocessing=None):
        self.feature = feature(audio_sampling_frequency,
                               movement_sampling_frequency)
        if preprocessing:
            self.preprocessing = preprocessing()
        else:
            self.preprocessing = None

    def fit(self, X, y=None):
        if self.preprocessing:
            self.preprocessing.fit(X, y)

    def transform(self, X, y=None):
        X_tfd = self.feature.transform(X, y)

        if self.preprocessing:
            X_sld = self.preprocessing.transform(X_tfd, y)
        else:
            X_sld = X_tfd

        return X_sld

    def fit_transform(self, X, y=None):
        X_tfd = self.feature.transform(X, y)

        if self.preprocessing:
            X_sld = self.preprocessing.fit_transform(X_tfd, y)
        else:
            X_sld = X_tfd

        return X_sld


class BaseFeature():
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        self.audio_sampling_frequency = audio_sampling_frequency
        self.movement_sampling_frequency = movement_sampling_frequency
