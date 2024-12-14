from chewbite_fusion.features.base import BaseFeature


class AccXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[1]) for window in file])

        return raw_movement


class AccYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[2]) for window in file])

        return raw_movement


class AccZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[3]) for window in file])

        return raw_movement


class GyrXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[4]) for window in file])

        return raw_movement


class GyrYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[5]) for window in file])

        return raw_movement


class GyrZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[6]) for window in file])

        return raw_movement


class MagXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[7]) for window in file])

        return raw_movement


class MagYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[8]) for window in file])

        return raw_movement


class MagZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[9]) for window in file])

        return raw_movement


class AccMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[10]) for window in file])

        return raw_movement


class GyrMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[11]) for window in file])

        return raw_movement


class MagMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for file in X:
            raw_movement.append([list(window[12]) for window in file])

        return raw_movement
