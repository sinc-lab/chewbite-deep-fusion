import random


class RandomEstimator:
    ''' Create a random classifier. '''
    def __init__(self):
        self.classes_ = None

    def fit(self, X, y):
        ''' Perform no training. '''
        self.classes_ = list(set(y))

    def predict(self, X):
        ''' Get random predictions based on available classes. '''
        y_pred = []
        for i in range(len(X)):
            y_pred.append(random.choice(self.classes_))

        return y_pred

    def clear_params(self):
        pass
