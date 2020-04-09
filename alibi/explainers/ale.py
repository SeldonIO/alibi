# flake8: noqa E131
import copy
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, TYPE_CHECKING, no_type_check

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ALE, DEFAULT_DATA_ALE

if TYPE_CHECKING:
    import matplotlib as mpl


class ALE(Explainer):

    def __init__(self,
                 predictor: Callable,
                 feature_names: List[str] = None,
                 target_names: List[str] = None):
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ALE))

        self.predictor = predictor
        self.feature_names = feature_names
        self.target_names = target_names

    def explain(self,
                X: np.ndarray,
                num_intervals: int = 40) -> Explanation:
        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional')
        n_features = X.shape[1]

        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]
        if self.target_names is None:
            pred = self.predictor(X[0].reshape(1, -1))
            n_targets = pred.shape[1]
            self.target_names = [f'c_{i}' for i in range(n_targets)]

        feature_values = []
        ale_values = []
        feature_deciles = []

        # TODO: use joblib to paralelise?
        for feature in range(n_features):
            q, ale = ale_num(self.predictor,
                             X=X,
                             feature=feature,
                             num_intervals=num_intervals)
            deciles = get_quantiles(X[:, feature])

            feature_values.append(q)
            ale_values.append(ale)
            feature_deciles.append(deciles)

        # TODO: an ALE plot requires a rugplot to gauge density of instances in each
        # feature region, should we calculate it here and return as part of the explanation
        # for further visualisation?
        return self.build_explanation(ale_values=ale_values,
                                      feature_values=feature_values,
                                      feature_deciles=feature_deciles)

    def build_explanation(self,
                          ale_values: List[np.ndarray],
                          feature_values: List[np.ndarray],
                          feature_deciles: List[np.ndarray]) -> Explanation:
        # TODO decide on the format for these lists of arrays
        # Currently each list element relates to a feature and each
        # column relates to an output dimension, this is different from e.g. SHAP

        data = copy.deepcopy(DEFAULT_DATA_ALE)
        data.update(ale_values=ale_values,
                    feature_values=feature_values,
                    feature_names=self.feature_names,
                    target_names=self.target_names,
                    feature_deciles=feature_deciles)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)


def get_quantiles(values: np.ndarray, num_points: int = 11, interpolation='linear') -> np.ndarray:
    """
    Calculate quantiles of values in an array.

    Parameters
    ----------
    values
        Array of values
    num_points
        Number of quantiles to calculate

    Returns
    -------
    Array of quantiles of the input values

    """
    percentiles = np.linspace(0, 100, num=num_points)
    quantiles = np.percentile(values, percentiles, axis=0, interpolation=interpolation)
    return quantiles


def ale_num(
        predict: Callable, X: np.ndarray, feature: int, num_intervals: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first order ALE for a numerical feature.

    Parameters
    ----------
    predict
        Model prediction function
    X
        Dataset to calculate the ALE on
    feature
        Index of the numerical feature for which to calculate ALE
    num_intervals
        Number of intervals to subdivide the feature into according to quantiles

    Returns
    -------
    q
        Array of quantiles of the input values, a num_intervals length vector
    ale
        ALE values for the feature, a num_intervals x n_outputs array

    """
    # TODO handle case when num_intervals is too large for the dataset
    num_points = num_intervals + 1
    q = np.unique(get_quantiles(X[:, feature], num_points=num_points))

    # find which interval each observation falls into
    indices = np.searchsorted(q, X[:, feature], side="left")
    indices[indices == 0] = 1  # put the smallest data point in the first interval
    interval_n = np.bincount(indices)  # number of points in each interval
    nonzero_interval_ix = np.argwhere(interval_n != 0).squeeze()
    nonzero_intervals = interval_n[nonzero_interval_ix]

    # predictions for the upper and lower ranges of intervals
    z_low = X.copy()
    z_high = X.copy()
    z_low[:, feature] = q[indices - 1]
    z_high[:, feature] = q[indices]
    p_low = predict(z_low)
    p_high = predict(z_high)

    # finite differences
    p_deltas = p_high - p_low

    # make a dataframe for averaging over intervals
    concat = np.column_stack((p_deltas, indices))
    df = pd.DataFrame(concat)

    # append dummy zero rows for intervals with no data points (to ensure lengths of vectors are consistent)
    # empty_intervals = np.argwhere(interval_n==0).reshape(-1, 1)
    # to_append = np.hstack((np.zeros((len(empty_intervals), df.shape[1] - 1)), empty_intervals))
    # df = df.append(pd.DataFrame(to_append), ignore_index=True)

    # TODO need to linearly interpolate instead of appending dummy zeros
    # TODO: alternatively, prune entries in q with no data points

    avg_p_deltas = df.groupby(df.shape[1] - 1).mean().values  # groupby indices

    # alternative numpy implementation (for 1-dimensional output only)
    # counts = np.bincount(indices, p_deltas)
    # avg_p_deltas = counts[1:] / interval_n[1:]

    # accummulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

    # pre-pend 0 for the left-most point
    zeros = np.zeros((1, accum_p_deltas.shape[1]))
    accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

    # mean effect
    ale0 = (((accum_p_deltas[:-1, :] + accum_p_deltas[1:, :]) / 2) * nonzero_intervals[:, np.newaxis]) \
               .sum(axis=0) / nonzero_intervals.sum()

    # center
    ale = accum_p_deltas - ale0

    # refine q
    q = np.hstack((q[[0]], q[nonzero_interval_ix]))

    return q, ale


@no_type_check
def _show_ale_num_mpl(exp: 'Explanation',
                      feature: int,
                      ax: 'mpl.axes.Axes' = None,
                      **kwargs) -> 'mpl.axes.Axes':
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    lines = ax.plot(exp.feature_values[feature], exp.ale_values[feature], '-o')

    # add decile markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.feature_deciles[feature][1:], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.feature_names[feature])
    ax.set_ylabel('ALE')
    ax.legend(lines, exp.target_names)

    return ax
