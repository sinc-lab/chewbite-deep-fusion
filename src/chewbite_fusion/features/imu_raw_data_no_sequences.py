from chewbite_fusion.features.base import BaseFeature


class AccXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[1])

        return raw_movement


class AccYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[2])

        return raw_movement


class AccZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[3])

        return raw_movement


class GyrXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[4])

        return raw_movement


class GyrYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[5])

        return raw_movement


class GyrZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[6])

        return raw_movement


class MagXRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[7])

        return raw_movement


class MagYRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[8])

        return raw_movement


class MagZRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[9])

        return raw_movement


class AccMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[10])

        return raw_movement


class GyrMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[11])

        return raw_movement


class MagMagnitudeVector(BaseFeature):
    def transform(self, X, y=None):
        raw_movement = []

        for window in X:
            raw_movement.append(window[12])

        return raw_movement
