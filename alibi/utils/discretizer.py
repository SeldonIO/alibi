import numpy as np
from typing import Dict, Callable, List, Union, Sequence
from functools import partial


class Discretizer(object):

    def __init__(self, data: np.ndarray, numerical_features: List[int], feature_names: List[str],
                 percentiles: Sequence[Union[int, float]] = (25, 50, 75)) -> None:
        """
        Initialize the discretizer

        Parameters
        ----------
        data
            Data to discretize
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names
        percentiles
            Percentiles used for discretization
        """

        self.to_discretize = numerical_features
        self.percentiles = percentiles

        bins = self.bins(data)
        bins = [np.unique(x) for x in bins]

        self.feature_intervals = {}  # type: Dict[int, list]
        self.lambdas = {}            # type: Dict[int, Callable] # TODO: Fix typing
        for feature, qts in zip(self.to_discretize, bins):

            # get nb of borders (nb of bins - 1) and the feature name
            n_bins = qts.shape[0]
            name = feature_names[feature]

            # create names for bins of discretized features
            self.feature_intervals[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.feature_intervals[feature].append('%.2f < %s <= %.2f' % (qts[i], name, qts[i + 1]))
            self.feature_intervals[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))
            self.lambdas[feature] = partial(self.get_percentiles, qts=qts)

    @staticmethod
    def get_percentiles(x: np.ndarray, qts: np.ndarray) -> np.ndarray:
        return np.searchsorted(qts, x)

    def bins(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Parameters
        ----------
        data
            Data to discretize

        Returns
        -------
        List with bin values for each feature that is discretized.
        """
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], self.percentiles))
            bins.append(qts)
        return bins

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data
            Data to discretize

        Returns
        -------
        Discretized version of data with the same dimension.
        """
        data_disc = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                data_disc[feature] = int(self.lambdas[feature](data_disc[feature]))
            else:
                data_disc[:, feature] = self.lambdas[feature](data_disc[:, feature]).astype(int)
        return data_disc
