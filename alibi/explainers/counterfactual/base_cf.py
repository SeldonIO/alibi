from scipy.spatial.distance import cityblock
import numpy as np
from abc import abstractmethod
from statsmodels import robust
from functools import reduce


def _flatten_X(X: np.array) -> np.array:
    if len(X.shape)>1:
        nb_features=reduce((lambda x, y: x * y), X.shape[1:])
        return X.reshape(X.shape[0], nb_features)
    else:
        return X


def _calculate_franges(X_train: np.array) -> list:
    X_train=_flatten_X(X_train)
    f_ranges = []
    print(X_train.shape)
    for i in range(X_train.shape[1]):
        mi, ma = X_train[:, i].min(), X_train[:, i].max()
        f_ranges.append((mi, ma))
    return f_ranges


def _mad_distance(x0: np.array, x1: np.array, mads: np.array) -> float:
    return (np.abs(x0-x1)/mads).sum()


class BaseCounterFactual(object):

    @abstractmethod
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.f_ranges = _calculate_franges(X_train)
        self.mads = robust.mad(X_train, axis=0)+10e-10

    def _metric_distance(self, x0, x1):
        if isinstance(self.callable_distance, str):
            if self.callable_distance=='l1_distance':
                self.callable_distance = cityblock
            elif self.callable_distance=='mad_distance':
                self.callable_distance = _mad_distance
            else:
                raise NameError('Metric {} not implemented. '
                                'For custom metrics, pass a callable function'.format(self.callable_distance))
        try:
            return self.callable_distance(x0, x1)
        except TypeError:
            return self.callable_distance(x0, x1, self.mads)
