import numpy as np
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
    ale = np.zeros(len(q) - 1)

    for i in range(len(q) - 1):
        # TODO this misses the data point equal to the upper quantile
        subset = X[(q[i] <= X[:, feature]) & (X[:, feature] < q[i + 1])]

        if len(subset) != 0:
            # TODO: batching for large datasets
            z_low = subset.copy()
            z_high = subset.copy()
            z_low[:, feature] = q[i]
            z_high[:, feature] = q[i + 1]

            ale[i] += (predict(z_high) - predict(z_low)).sum() / len(subset)

    ale = ale.cumsum()
    ale -= ale.mean()

    return q, ale


def show_first_ale_num_altair(
    X: np.ndarray, q: np.ndarray, ale: np.ndarray, feature_name: str = None
) -> None:
    import altair as alt
    import pandas as pd

    if feature_name is None:
        feature_name = "feature"
    data = pd.DataFrame(
        {
            feature_name: (q[1:] + q[:-1])
            / 2,  # interpolate between quantile midpoints
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
