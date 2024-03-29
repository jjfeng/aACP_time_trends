import numpy as np
from numpy import ndarray


class Dataset:
    """
    Stores data
    """

    def __init__(self, x: ndarray = None, y: ndarray = None, num_classes: int = None):
        """
        @param x: np array of covariates (each row is observation)
        @param y: column vector (N X 1) of responses
        @param num_classes: number of classes, if a multinomial. otherwise this is None
        """
        self.x = x
        self.y = y
        self.num_classes = num_classes

    @property
    def num_obs(self):
        return self.x.shape[0]

    def merge(self, dataset):
        return Dataset(
            np.vstack((self.x, dataset.x)),
            np.vstack((self.y, dataset.y)),
            self.num_classes,
        )

    def subset(self, idxs):
        return Dataset(self.x[idxs], self.y[idxs], self.num_classes)

    def get_train(self, holdout_frac: float):
        num_train = int(self.num_obs * (1 -
            holdout_frac))
        idxs = np.arange(num_train)
        return Dataset(self.x[idxs], self.y[idxs], self.num_classes)

    def get_holdout(self, holdout_frac: float):
        num_train = int(self.num_obs * (1 -
            holdout_frac))
        idxs = np.arange(num_train, self.num_obs)
        return Dataset(self.x[idxs], self.y[idxs], self.num_classes)
