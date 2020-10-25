import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from proposer import Proposer
from trial_data import TrialData


class RandomForestWrap(RandomForestClassifier):
    def _criterion(self, pred, y):
        # hinge loss
        y = y.flatten()
        margin = (np.sign(y - 0.5) * 4 * (pred - 0.5)).astype(float)
        return np.maximum(0, 1 - margin) / 3

    def predict(self, X, t=None):
        return super().predict_proba(X)[:, 1]

    def loss(self, dataset):
        pred = self.predict_proba(dataset.x)
        return self._criterion(pred, dataset.y)


class RandomForestRWrap(RandomForestRegressor):
    def _criterion(self, pred, y):
        # hinge loss
        y = y.flatten()
        return np.abs(y - pred)

    def predict(self, X, t=None):
        return super().predict(X)

    def loss(self, dataset):
        pred = self.predict(dataset.x)
        return self._criterion(pred, dataset.y)
