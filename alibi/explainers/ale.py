# flake8: noqa E131
import numpy as np
import pandas as pd
from typing import Callable, Tuple


def get_quantiles(values: np.ndarray, num_points: int = 11) -> np.ndarray:
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
    quantiles = np.percentile(values, percentiles, axis=0)
    return quantiles


def first_ale_num(
        predict: Callable, X: np.ndarray, feature: int, num_points: int = 11
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
    num_points
        Number of quantiles to calculate for the feature

    Returns
    -------
    q
        Array of quantiles of the input values
    ale
        ALE values for the feature

    """
    # TODO the following only works for regression
    q = get_quantiles(X[:, feature], num_points=num_points)
    # ale = np.zeros(len(q) - 1)  # 1 less interval than quantiles
    # subset_lengths = np.zeros(len(q) - 1)

    indices = np.searchsorted(q, X[:, feature], side="left")
    indices[indices == 0] = 1  # put the smallest data point in the first interval

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
    df = pd.DataFrame({'interval': indices, 'p_delta': p_deltas})
    avg_p_deltas = df.groupby('interval').mean()

    # accummulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas)
    # TODO: why pre-pend 0?
    accum_p_deltas.loc[0] = 0
    accum_p_deltas.index = accum_p_deltas.index + 1
    accum_p_deltas.sort_index(inplace=True)

    interval_n = np.bincount(indices)

    # mean effect
    apd = accum_p_deltas.p_delta.values
    ale0 = (((apd[:-1] + apd[1:]) / 2) * interval_n[1:]).sum() / interval_n.sum()

    # center
    ale = apd - ale0

    # or i in range(1, num_points):
    #   if i == num_points - 1:
    #       subset = X[indices >= i]  # merge last point into the same quantile
    #   else:
    #       subset = X[indices == i]
    #   # TODO this misses the data point equal to the upper quantile
    #   # subset = X[(q[i] <= X[:, feature]) & (X[:, feature] < q[i + 1])]
    #   subset_lengths[i - 1] = len(subset)

    #   if len(subset) != 0:
    #       # TODO: batching for large datasets
    #       z_low = subset.copy()
    #       z_high = subset.copy()
    #       z_low[:, feature] = q[i - 1]
    #       z_high[:, feature] = q[i]

    #       ale[i - 1] += (predict(z_high) - predict(z_low)).sum() / len(subset)
    # ale[i] += ((predict(z_high) - predict(z_low)) / (z_high[:, feature] - z_low[:, feature])).sum()

    # ale = ale.cumsum()
    # ale -= np.dot(ale, subset_lengths) / X.shape[0]
    return q, ale


def show_first_ale_num_altair(
        X: np.ndarray, q: np.ndarray, ale: np.ndarray, feature_name: str = None, center: bool = True
) -> None:
    import altair as alt
    import pandas as pd

    if feature_name is None:
        feature_name = "feature"
    if center:
        # features = (q[1:] + q[:-1]) / 2  # interpolate between quantile midpoints
        features = q
    else:
        pass  # TODO: original definition is a step function, or is it? c.f. integral formulation Molnar

    data = pd.DataFrame(
        {
            feature_name: features,
            "ale": ale,
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
            .encode(alt.X("0:Q", title=feature_name), tooltip="0:Q")
    )

    (base + x_ticks).display()


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    X, y = load_boston(return_X_y=True)
    lr = LinearRegression().fit(X, y)

    q, ale = first_ale_num(lr.predict, X, 0, num_points=11)
