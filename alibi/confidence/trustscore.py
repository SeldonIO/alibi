import logging
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier

logger = logging.getLogger(__name__)


class TrustScore:

    def __init__(self,
                 k_filter: int = 10,
                 alpha: float = 0.,
                 filter_type: Optional[str] = None,
                 leaf_size: int = 40,
                 metric: str = 'euclidean',
                 dist_filter_type: str = 'point') -> None:
        """
        Initialize trust scores.

        Parameters
        ----------
        k_filter
            Number of neighbors used during either kNN distance or probability filtering.
        alpha
            Fraction of instances to filter out to reduce impact of outliers.
        filter_type
            Filter method: ``'distance_knn'`` | ``'probability_knn'``.
        leaf_size
            Number of points at which to switch to brute-force. Affects speed and memory required to build trees.
            Memory to store the tree scales with `n_samples / leaf_size`.
        metric
            Distance metric used for the tree. See `sklearn` DistanceMetric class for a list of available metrics.
        dist_filter_type
            Use either the distance to the k-nearest point (``dist_filter_type = 'point'``) or
            the average distance from the first to the k-nearest point in the data (``dist_filter_type = 'mean'``).
        """
        self.k_filter = k_filter
        self.alpha = alpha
        self.filter = filter_type
        self.eps = 1e-12
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_filter_type = dist_filter_type

    def filter_by_distance_knn(self, X: np.ndarray) -> np.ndarray:
        """
        Filter out instances with low kNN density. Calculate distance to k-nearest point in the data for each
        instance and remove instances above a cutoff distance.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Filtered data.
        """
        kdtree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        knn_r = kdtree.query(X, k=self.k_filter + 1)[0]  # distances from 0 to k-nearest points
        if self.dist_filter_type == 'point':
            knn_r = knn_r[:, -1]
        elif self.dist_filter_type == 'mean':
            knn_r = np.mean(knn_r[:, 1:], axis=1)  # exclude distance of instance to itself
        cutoff_r = np.percentile(knn_r, (1 - self.alpha) * 100)  # cutoff distance
        X_keep = X[np.where(knn_r <= cutoff_r)[0], :]  # define instances to keep
        return X_keep

    def filter_by_probability_knn(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out instances with high label disagreement amongst its k nearest neighbors.

        Parameters
        ----------
        X
            Data.
        Y
            Predicted class labels.

        Returns
        -------
        Filtered data and labels.
        """
        if self.k_filter == 1:
            logger.warning('Number of nearest neighbors used for probability density filtering should '
                           'be >1, otherwise the prediction probabilities are either 0 or 1 making '
                           'probability filtering useless.')
        # fit kNN classifier and make predictions on X
        clf = KNeighborsClassifier(n_neighbors=self.k_filter, leaf_size=self.leaf_size, metric=self.metric)
        clf.fit(X, Y)
        preds_proba = clf.predict_proba(X)
        # define cutoff and instances to keep
        preds_max = np.max(preds_proba, axis=1)
        cutoff_proba = np.percentile(preds_max, self.alpha * 100)  # cutoff probability
        keep_id = np.where(preds_max >= cutoff_proba)[0]  # define id's of instances to keep
        X_keep, Y_keep = X[keep_id, :], Y[keep_id]
        return X_keep, Y_keep

    def fit(self, X: np.ndarray, Y: np.ndarray, classes: Optional[int] = None) -> None:
        """
        Build KDTrees for each prediction class.

        Parameters
        ----------
        X
            Data.
        Y
            Target labels, either one-hot encoded or the actual class label.
        classes
            Number of prediction classes, needs to be provided if `Y` equals the predicted class.
        """
        self.classes = classes if classes is not None else Y.shape[1]
        self.kdtrees = [None] * self.classes  # type: Any
        self.X_kdtree = [None] * self.classes  # type: Any

        # KDTree and kNeighborsClassifier need 2D data
        if len(X.shape) > 2:
            logger.warning('Reshaping data from {0} to {1} so k-d trees can '
                           'be built.'.format(X.shape, X.reshape(X.shape[0], -1).shape))
            X = X.reshape(X.shape[0], -1)

        # make sure Y represents predicted classes, not one-hot encodings
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)  # type: ignore

        if self.filter == 'probability_knn':
            X_filter, Y_filter = self.filter_by_probability_knn(X, Y)

        for c in range(self.classes):

            if self.filter is None:
                X_fit = X[np.where(Y == c)[0]]
            elif self.filter == 'distance_knn':
                X_fit = self.filter_by_distance_knn(X[np.where(Y == c)[0]])
            elif self.filter == 'probability_knn':
                X_fit = X_filter[np.where(Y_filter == c)[0]]

            no_x_fit = len(X_fit) == 0
            if no_x_fit and len(X[np.where(Y == c)[0]]) == 0:
                logger.warning('No instances available for class %s', c)
            elif no_x_fit:
                logger.warning('Filtered all the instances for class %s. Lower alpha or check data.', c)

            self.kdtrees[c] = KDTree(X_fit, leaf_size=self.leaf_size, metric=self.metric)  # build KDTree for class c
            self.X_kdtree[c] = X_fit

    def score(self, X: np.ndarray, Y: np.ndarray, k: int = 2, dist_type: str = 'point') \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate trust scores = ratio of distance to closest class other than the
        predicted class to distance to predicted class.

        Parameters
        ----------
        X
            Instances to calculate trust score for.
        Y
            Either prediction probabilities for each class or the predicted class.
        k
            Number of nearest neighbors used for distance calculation.
        dist_type
            Use either the distance to the k-nearest point (``dist_type = 'point'``) or
            the average distance from the first to the k-nearest point in the data (``dist_type = 'mean'``).

        Returns
        -------
        Batch with trust scores and the closest not predicted class.
        """
        # make sure Y represents predicted classes, not probabilities
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)  # type: ignore

        # KDTree needs 2D data
        if len(X.shape) > 2:
            logger.warning('Reshaping data from {0} to {1} so k-d trees can '
                           'be queried.'.format(X.shape, X.reshape(X.shape[0], -1).shape))
            X = X.reshape(X.shape[0], -1)

        # init distance matrix: [nb instances, nb classes]
        d = np.tile(np.nan, (X.shape[0], self.classes))  # type: np.ndarray

        for c in range(self.classes):
            d_tmp = self.kdtrees[c].query(X, k=k)[0]  # get k nearest neighbors for each class
            if dist_type == 'point':
                d[:, c] = d_tmp[:, -1]
            elif dist_type == 'mean':
                d[:, c] = np.mean(d_tmp, axis=1)

        sorted_d = np.sort(d, axis=1)  # sort distance each instance in batch over classes
        # get distance to predicted and closest other class and calculate trust score
        d_to_pred = d[range(d.shape[0]), Y]
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
        trust_score = d_to_closest_not_pred / (d_to_pred + self.eps)
        # closest not predicted class
        class_closest_not_pred = np.where(d == d_to_closest_not_pred.reshape(-1, 1))[1]
        return trust_score, class_closest_not_pred
