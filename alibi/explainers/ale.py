# flake8: noqa E131
import copy
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ALE, DEFAULT_DATA_ALE


class ALE(Explainer):

    def __init__(self,
                 predictor: Callable):
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ALE))

        self.predictor = predictor

    def explain(self,
                X: np.ndarray,
                num_intervals: int = 40) -> Explanation:
        # TODO raise an error if X is not 2D
        n_features = X.shape[1]
        feature_values = []
        ale_values = []

        for feature in range(n_features):
            q, ale = ale_num(self.predictor,
                             X=X,
                             feature=feature,
                             num_intervals=num_intervals)

            feature_values.append(q)
            ale_values.append(ale)

        # TODO: an ALE plot requires a rugplot to gauge density of instances in each
        # feature region, should we calculate it here and return as part of the explanation
        # for further visualisation?
        return self.build_explanation(ale_values=ale_values, feature_values=feature_values)

    def build_explanation(self,
                          ale_values: List[np.ndarray],
                          feature_values: List[np.ndarray]) -> Explanation:
        # TODO decide on the format for these lists of arrays
        # Currently each list element relates to a feature and each
        # column relates to an output dimension, this is different from e.g. SHAP

        data = copy.deepcopy(DEFAULT_DATA_ALE)
        data.update(ale_values=ale_values,
                    feature_values=feature_values,
                    feature_names=None)

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
    avg_p_deltas = df.groupby(df.shape[1] - 1).mean().values  # groupby indices

    # alternative numpy implementation
    # counts = np.bincount(indices, p_deltas)
    # avg_p_deltas = counts[1:] / interval_n[1:]

    # accummulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

    # pre-pend 0 for the left-most point
    zeros = np.zeros((1, accum_p_deltas.shape[1]))
    accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

    # mean effect
    ale0 = (((accum_p_deltas[:-1, :] + accum_p_deltas[1:, :]) / 2) * interval_n[1:, np.newaxis]) \
               .sum(axis=0) / interval_n.sum()

    # center
    ale = accum_p_deltas - ale0

    return q, ale


def show_first_ale_num_altair(X: np.ndarray, q: np.ndarray, ale: np.ndarray, feature: int,
                              feature_name: str = None) -> None:
    import altair as alt
    import pandas as pd

    if feature_name is None:
        feature_name = "feature"

    data = pd.DataFrame(
        {
            feature_name: q,
            "ale": ale[:, 0],  # TODO: handle multiclass
        }
    )

    height = 250
    tick_height = 20
    tick_baseline = height - tick_height / 2

    # ALE plot
    base = (
        alt.Chart(data, height=height)
            .mark_line(point=True)
            .encode(x=feature_name, y="ale", tooltip=["ale", feature_name])
    )

    # rug plot
    pdX = pd.DataFrame(X)
    x_ticks = (
        alt.Chart(pdX)
            .mark_tick(size=tick_height, y=tick_baseline)
            .encode(alt.X("{}:Q".format(feature), title=feature_name), tooltip="{}:Q".format(feature))
    )

    (base + x_ticks).display()


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    X, y = load_boston(return_X_y=True)
    lr = LinearRegression().fit(X, y)

    q, ale = ale_num(lr.predict, X, 0, num_intervals=10)
