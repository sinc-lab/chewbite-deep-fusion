from chewbite_fusion.features.base import BaseFeature


class AccXDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[1][1:] - w[1][:-1]) for w in file])

        return raw_data


class AccYDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[2][1:] - w[2][:-1]) for w in file])

        return raw_data


class AccZDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[3][1:] - w[3][:-1]) for w in file])

        return raw_data


class GyrXDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[4][1:] - w[4][:-1]) for w in file])

        return raw_data


class GyrYDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[5][1:] - w[5][:-1]) for w in file])

        return raw_data


class GyrZDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[6][1:] - w[6][:-1]) for w in file])

        return raw_data


class MagXDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[7][1:] - w[7][:-1]) for w in file])

        return raw_data


class MagYDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[8][1:] - w[8][:-1]) for w in file])

        return raw_data


class MagZDiff(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[9][1:] - w[9][:-1]) for w in file])

        return raw_data


class AccMagnitudeDiffVector(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[10][1:] - w[10][:-1]) for w in file])

        return raw_data


class GyrMagnitudeDiffVector(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[11][1:] - w[11][:-1]) for w in file])

        return raw_data


class MagMagnitudeDiffVector(BaseFeature):
    def transform(self, X, y=None):
        raw_data = []

        for file in X:
            raw_data.append([[0] + list(w[12][1:] - w[12][:-1]) for w in file])

        return raw_data
