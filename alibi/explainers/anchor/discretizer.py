import numpy as np
from typing import Dict, Callable, List


class Discretizer(object):

    def __init__(self, data: np.ndarray, categorical_features: List[int], feature_names: List[str],
                 percentiles: List[int] = [25, 50, 75]) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize
        categorical_features
            List of indices corresponding to the categorical columns. These features will not be discretized.
            The other features will be considered continuous and therefore discretized.
        feature_names
            List with feature names
        percentiles
            Percentiles used for discretization
        """
        self.to_discretize = ([x for x in range(data.shape[1]) if x not in categorical_features])
        self.percentiles = percentiles

        bins = self.bins(data)
        bins = [np.unique(x) for x in bins]

        self.names = {}  # type: Dict[int, list]
        self.lambdas = {}  # type: Dict[int, Callable]
        for feature, qts in zip(self.to_discretize, bins):
            # get nb of borders (nb of bins - 1) and the feature name
            n_bins = qts.shape[0]
            name = feature_names[feature]

            # create names for bins of discretized features
            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' % (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))
            self.lambdas[feature] = lambda x, qts = qts: np.searchsorted(qts, x)

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
