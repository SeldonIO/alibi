import sys
import copy
import logging

import inspect
import numpy as np
from enum import Enum
from alibi.api.interfaces import Explainer
from alibi.api.defaults import DEFAULT_META_PERMUTATION_IMPORTANCE, DEFAULT_DATA_PERMUTATION_IMPORTANCE
from typing import Callable, Optional, Union, List, Dict
from tqdm import tqdm

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

logger = logging.getLogger(__name__)


class Method(str, Enum):
    """ Enumeration of supported method. """
    EXACT = 'exact'
    ESTIMATE = 'estimate'


class Kind(str, Enum):
    """ Enumeration of supported kind. """
    DIFFERENCE = 'difference'
    RATIO = 'ratio'


class PermutationImportance(Explainer):
    def __init__(self,
                 predictor: Callable[[np.array], np.array],
                 feature_names: Optional[List[str]] = None,
                 verbose: bool = False):

        super().__init__(meta=copy.deepcopy(DEFAULT_META_PERMUTATION_IMPORTANCE))
        self.predictor = predictor
        self.feature_names = feature_names
        self.verbose = verbose

    def explain(self,
                X: np.ndarray,
                y: np.ndarray,
                loss_fns: Union
                    [
                      Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                      Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ],
                features: Optional[List[int]] = None,
                method: Literal["estimate", "exact"] = "estimate",
                kind: Literal["ratio", "difference"] = "ratio",
                n_repeats: int = 50,
                sample_weight: Optional[np.ndarray] = None):

        n_features = X.shape[1]

        # set the `features_names` when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # construct `feature_names` based on the `features`. If `features` is ``None``, then initialize
        # `features` with all single feature available in the dataset.
        if features:
            feature_names = [self.feature_names[f] for f in features]
        else:
            feature_names = self.feature_names
            features = list(range(n_features))

        # compute the base score
        loss_orig = {}

        for loss_name, loss_fn in loss_fns.items():
            loss_orig[loss_name] = self._compute_loss(y_true=y,
                                                      y_pred=self.predictor(X),
                                                      loss_fn=loss_fn,
                                                      sample_weight=sample_weight)


        # compute permutation feature importance for every feature
        # TODO: implement parallel version - future work as it can be done for ALE too
        perm_importance = []

        for ifeature in tqdm(features, disable=not self.verbose):
            perm_importance.append(
                self._compute_permutation_importance(
                    X=X,
                    y=y,
                    loss_fns=loss_fns,
                    method=method,
                    kind=kind,
                    n_repeats=n_repeats,
                    sample_weight=sample_weight,
                    feature=ifeature,
                    loss_orig=loss_orig,
                )
            )
        return perm_importance

    def _compute_loss(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      loss_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                      sample_weight: Optional[np.ndarray]):

        # get scoring function arguments
        args = inspect.getfullargspec(loss_fn).args

        if 'y_true' not in args:
            raise ValueError('The `scoring` function must have the argument `y_true` in its definition.')

        if 'y_pred' not in args:
            raise ValueError('The `scoring` function must have the argument `y_pred` in its definition.')

        if 'sample_weight' not in args:
            # some scores might not support `sample_weight` such as:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
            if sample_weight is not None:
                logger.warning(f"The loss function '{loss_name}' does not support argument `sample_weight`. "
                               f"Calling the method without `sample_weight`.")

            return loss_fn(y_true=y_true, y_pred=y_pred)

        # call scoring function with all parameters.
        return loss_fn(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def _compute_permutation_importance(self,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                                        method: Literal["estimate", "exact"],
                                        kind: Literal["difference", "ratio"],
                                        n_repeats: int,
                                        sample_weight: Optional[np.ndarray],
                                        feature: int,
                                        loss_orig: Dict[str, float]):

        if method == Method.EXACT:
            # computation of the exact statistic which is quadratic in the number of samples
            return self._compute_exact(X=X,
                                       y=y,
                                       feature=feature,
                                       loss_fns=loss_fns,
                                       kind=kind,
                                       sample_weight=sample_weight,
                                       loss_orig=loss_orig)
        # sample approximation
        return self._compute_estimate(X=X,
                                      y=y,
                                      feature=feature,
                                      loss_fns=loss_fns,
                                      kind=kind,
                                      n_repeats=n_repeats,
                                      sample_weight=sample_weight,
                                      loss_orig=loss_orig)

    def _compute_exact(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       feature: int,
                       loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                       kind: str,
                       sample_weight: Optional[np.ndarray],
                       loss_orig: Dict[str, float]):
        y_pred = []
        weights = [] if sample_weight else None
        loss_permuted = {}

        for i in range(len(X)):
            # create dataset
            X_tmp = np.tile(X[i:i+1], reps=(len(X) - 1, 1))
            X_tmp[:, feature] = np.delete(arr=X[:, feature], obj=i, axis=0)

            # compute predictions
            y_pred.append(self.predictor(X_tmp))

            # create sample weights if necessary
            if sample_weight is not None:
                weights.append(np.full(shape=(len(X_tmp),), fill_value=sample_weight[i]))

        # concatenate all predictions and construct ground-truth array. At this point, the `y_pre` vector
        # should contain `N x (N - 1)` predictions, where `N` is the number of samples in `X`.
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.tile(y.reshape(-1, 1), reps=(1, len(X) - 1)).reshape(-1)

        for loss_name, loss_fn in loss_fns.items():
            loss_permuted[loss_name] = self._compute_loss(y_true=y_true,
                                                          y_pred=y_pred,
                                                          sample_weight=weights,
                                                          loss_fn=loss_fn)

        return {loss_name: self._compute_importance(
            loss_orig=loss_orig[loss_name],
            loss_permuted=loss_permuted[loss_name],
            kind=kind
        ) for loss_name in loss_fns}

    def _compute_estimate(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          feature: int,
                          loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                          kind: str,
                          n_repeats: int,
                          sample_weight: Optional[np.ndarray],
                          loss_orig: Dict[str, float]):

        N = len(X)
        start, middle, end = 0, N // 2, N if N % 2 == 0 else N - 1
        fh, sh = np.s_[start:middle], np.s_[middle:end]
        loss_permuted = {loss_name: [] for loss_name in loss_fns}

        for i in range(n_repeats):
            # get random permutation. Note that this includes also the last element
            shuffled_indices = np.random.permutation(len(X))

            # shuffle the dataset
            X_tmp, y_tmp = X[shuffled_indices].copy(), y[shuffled_indices].copy()
            sample_weight_tmp = None if (sample_weight is None) else sample_weight[shuffled_indices].copy()

            # permute values from the first half into the second half and the other way around
            fvals_tmp = X_tmp[fh, feature].copy()
            X_tmp[fh, feature] = X_tmp[sh, feature]
            X_tmp[sh, feature] = fvals_tmp

            # compute scores
            y_pred = self.predictor(X_tmp[:end])
            y_true = y_tmp[:end]
            weights = None if (sample_weight_tmp is None) else sample_weight_tmp[:end]

            for loss_name, loss_fn in loss_fns.items():
                loss_permuted[loss_name].append(
                    self._compute_loss(y_true=y_true,
                                       y_pred=y_pred,
                                       sample_weight=weights,
                                       loss_fn=loss_fn)
                )

        importance = {}

        for loss_name in loss_fns:
            importance_values = [
                self._compute_importance(
                    loss_orig=loss_orig[loss_name],
                    loss_permuted=loss_permuted_value,
                    kind=kind,
                ) for loss_permuted_value in loss_permuted[loss_name]]

            importance[loss_name] = {
                "mean": np.mean(importance_values),
                "std": np.std(importance_values)
            }

        return importance

    def _compute_importance(self, loss_orig: float, loss_permuted: float, kind):
        return loss_permuted / loss_orig if kind == Kind.RATIO else loss_permuted - loss_orig


