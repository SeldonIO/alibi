# flake8: noqa E131
import copy
import math
import numpy as np
import pandas as pd
from itertools import count
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING, no_type_check
from typing_extensions import Literal

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ALE, DEFAULT_DATA_ALE

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class ALE(Explainer):

    def __init__(self,
                 predictor: Callable,
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None) -> None:
        """
        Accumulated Local Effects for tabular datasets. Current implementation supports first order
        feature effects of numerical features.

        Parameters
        ----------
        predictor
            A callable that takes in an NxF array as input and outputs an NxT array (N - number of
            data points, F - number of features, T - number of outputs/targets (e.g. 1 for single output
            regression, >=2 for classification).
        feature_names
            A list of feature names used for displaying results.
        target_names
            A list of target/output names used for displaying results.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ALE))

        self.predictor = predictor
        self.feature_names = feature_names
        self.target_names = target_names

    def explain(self, X: np.ndarray, min_bin_points: int = 4) -> Explanation:
        """
        Calculate the ALE curves for each feature with respect to the dataset `X`.

        Parameters
        ----------
        X
            An NxF tabular dataset used to calculate the ALE curves. This is typically the training dataset
            or a representative sample.
        min_bin_points
            Minimum number of points each discretized interval should contain to ensure more precise
            ALE estimation.

        Returns
        -------
            An `Explanation` object containing the data and the metadata of the calculated ALE curves.

        """
        self.meta['params'].update(min_bin_points=min_bin_points)

        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional')
        n_features = X.shape[1]

        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]
        if self.target_names is None:
            pred = np.atleast_2d(self.predictor(X[0].reshape(1, -1)))
            n_targets = pred.shape[1]
            self.target_names = [f'c_{i}' for i in range(n_targets)]
        self.feature_names = np.array(self.feature_names)
        self.target_names = np.array(self.target_names)

        feature_values = []
        ale_values = []
        ale0 = []
        feature_deciles = []

        # TODO: use joblib to paralelise?
        for feature in range(n_features):
            q, ale, a0 = ale_num(
                self.predictor,
                X=X,
                feature=feature,
                min_bin_points=min_bin_points
            )
            deciles = get_quantiles(X[:, feature], num_quantiles=11)

            feature_values.append(q)
            ale_values.append(ale)
            ale0.append(a0)
            feature_deciles.append(deciles)

        constant_value = self.predictor(X).mean()
        # TODO: an ALE plot ideally requires a rugplot to gauge density of instances in the feature space.
        # I've replaced this with feature deciles which is coarser but has constant space complexity
        # as opposed to a rugplot. Alternatively, could consider subsampling to produce a rug with some
        # maximum number of points.
        return self.build_explanation(
            ale_values=ale_values,
            ale0=ale0,
            constant_value=constant_value,
            feature_values=feature_values,
            feature_deciles=feature_deciles
        )

    def build_explanation(self,
                          ale_values: List[np.ndarray],
                          ale0: List[np.ndarray],
                          constant_value: float,
                          feature_values: List[np.ndarray],
                          feature_deciles: List[np.ndarray]) -> Explanation:
        """
        Helper method to build the Explanation object.
        """
        # TODO decide on the format for these lists of arrays
        # Currently each list element relates to a feature and each column relates to an output dimension,
        # this is different from e.g. SHAP but arguably more convenient for ALE.

        data = copy.deepcopy(DEFAULT_DATA_ALE)
        data.update(
            ale_values=ale_values,
            ale0=ale0,
            constant_value=constant_value,
            feature_values=feature_values,
            feature_names=self.feature_names,
            target_names=self.target_names,
            feature_deciles=feature_deciles
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)


def get_quantiles(values: np.ndarray, num_quantiles: int = 11, interpolation='linear') -> np.ndarray:
    """
    Calculate quantiles of values in an array.

    Parameters
    ----------
    values
        Array of values.
    num_quantiles
        Number of quantiles to calculate.

    Returns
    -------
    Array of quantiles of the input values.

    """
    percentiles = np.linspace(0, 100, num=num_quantiles)
    quantiles = np.percentile(values, percentiles, axis=0, interpolation=interpolation)
    return quantiles


def bisect_fun(fun: Callable, target: float, lo: int, hi: int) -> int:
    """
    Bisection algorithm for function evaluation with integer support.

    Assumes the function is non-decreasing on the interval [lo, hi].
    Return an integer value v such that for all x<v, fun(x)<target and for all x>=v fun(x)>=target.
    This is equivalent to the library function `bisect.bisect_left` but for functions defined on integers.

    Parameters
    ----------
    fun
        A function defined on integers in the range [lo, hi] and returning floats.
    target
        Target value to be searched for.
    lo
        Lower bound of the domain.
    hi
        Upper bound of the domain.

    Returns
    -------
    Integer index.

    """
    while lo < hi:
        mid = (lo + hi) // 2
        if fun(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def minimum_satisfied(values: np.ndarray, min_bin_points: int, n: int) -> int:
    """
    Calculates whether the partition into bins induced by n quantiles
    has the minimum number of points in each resulting bin.

    Parameters
    ----------
    values
        Array of feature values.
    min_bin_points
        Minimum number of points each discretized interval needs to contain.
    n
        Number of quantiles.

    Returns
    -------
        Integer encoded boolean with 1 - each bin has at least `min_bin_points` and 0 otherwise.
    """
    q = np.unique(get_quantiles(values, num_quantiles=n))
    indices = np.searchsorted(q, values, side='left')
    indices[indices == 0] = 1
    interval_n = np.bincount(indices)
    return int(np.all(interval_n[1:] > min_bin_points))


def adaptive_grid(values: np.ndarray, min_bin_points: int = 1) -> Tuple[np.ndarray, int]:
    """
    Find the optimal number of quantiles for the range of values so that each resulting bin
    contains at least `min_bin_points`. Uses bisection.

    Parameters
    ----------
    values
        Array of feature values.
    min_bin_points
        Minimum number of points each discretized interval should contain to ensure more precise
        ALE estimation.

    Returns
    -------
    q
        Unique quantiles.
    num_quantiles
        Number of non-unique quantiles the feature array was subdivided into.

    Notes
    -----
    This is a heuristic procedure since the bisection algorithm is applied
    to a function which is not monotonic. This will not necessarily find the
    maximum number of bins the interval can be subdivided into to satisfy
    the minimum number of points in each resulting bin.
    """

    # function to bisect
    def minimum_not_satisfied(values: np.ndarray, min_bin_points: int, n: int) -> int:
        """
        Logical not of `minimum_satisfied`, see function for parameter information.
        """
        return 1 - minimum_satisfied(values, min_bin_points, n)

    fun = partial(minimum_not_satisfied, values, min_bin_points)

    # bisect
    num_quantiles = bisect_fun(fun=fun, target=0.5, lo=0, hi=len(values)) - 1
    q = np.unique(get_quantiles(values, num_quantiles=num_quantiles))

    return q, num_quantiles


def ale_num(
        predictor: Callable,
        X: np.ndarray,
        feature: int,
        min_bin_points: int = 4) -> Tuple[np.ndarray, ...]:
    """
    Calculate the first order ALE curve for a numerical feature.

    Parameters
    ----------
    predictor
        Model prediction function.
    X
        Dataset for which ALE curves are computed.
    feature
        Index of the numerical feature for which to calculate ALE.
    min_bin_points
        Minimum number of points each discretized interval should contain to ensure more precise
        ALE estimation.

    Returns
    -------
    q
        Array of quantiles of the input values.
    ale
        ALE values for each feature at each of the points in q.
    ale0
        The constant offset used to center the ALE curves.

    """
    q, _ = adaptive_grid(X[:, feature], min_bin_points)

    # find which interval each observation falls into
    indices = np.searchsorted(q, X[:, feature], side="left")
    indices[indices == 0] = 1  # put the smallest data point in the first interval
    interval_n = np.bincount(indices)  # number of points in each interval

    # predictions for the upper and lower ranges of intervals
    z_low = X.copy()
    z_high = X.copy()
    z_low[:, feature] = q[indices - 1]
    z_high[:, feature] = q[indices]
    p_low = predictor(z_low)
    p_high = predictor(z_high)

    # finite differences
    p_deltas = p_high - p_low

    # make a dataframe for averaging over intervals
    concat = np.column_stack((p_deltas, indices))
    df = pd.DataFrame(concat)

    avg_p_deltas = df.groupby(df.shape[1] - 1).mean().values  # groupby indices

    # accummulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

    # pre-pend 0 for the left-most point
    zeros = np.zeros((1, accum_p_deltas.shape[1]))
    accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

    # mean effect, R's `ALEPlot` and `iml` version (approximation per interval)
    ale0 = (0.5 * (accum_p_deltas[:-1, :] + accum_p_deltas[1:, :]) * interval_n[1:, np.newaxis]).sum(axis=0)
    ale0 = ale0 / interval_n.sum()

    # crude approximation (assume datapoints on interval endpoints)
    # ale0 = accum_p_deltas.mean(axis=0)

    # exact marginalisation
    # exact_ale = accum_p_deltas[indices - 1] + ((X[:, feature] - q[indices])) / (q[indices] - q[indices - 1]) * (
    #            accum_p_deltas[indices] - accum_p_deltas[indices - 1])
    # ale0 = exact_ale.mean()

    # center
    ale = accum_p_deltas - ale0

    return q, ale, ale0


# no_type_check is needed because exp is a generic explanation and so mypy doesn't know that the
# attributes actually exist... As a side effect the type information does not show up in the static
# docs. Will need to re-think this.
@no_type_check
def plot_ale(exp: Explanation,
             features: Union[List[Union[int, str]], Literal['all']] = 'all',
             targets: Union[List[Union[int, str]], Literal['all']] = 'all',
             n_cols: int = 3,
             sharey: str = 'all',
             constant: bool = False,
             ax: Union['plt.Axes', np.ndarray, None] = None,
             line_kw: Optional[dict] = None,
             fig_kw: Optional[dict] = None) -> 'np.ndarray':
    """
    Plot ALE curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the `ALE.explain` method.
    features
        A list of features for which to plot the ALE curves or `all` for all features.
        Can be a mix of integers denoting feature index or strings denoting entries in
        `exp.feature_names`. Defaults to 'all'.
    targets
        A list of targets for which to plot the ALE curves or `all` for all targets.
        Can be a mix of integers denoting target index or strings denoting entries in
        `exp.target_names`. Defaults to 'all'.
    n_cols
        Number of columns to organize the resulting plot into.
    sharey
        A parameter specifying whether the y-axis of the ALE curves should be on the same scale
        for several features. Possible values are `all`, `row`, `None`.
    constant
        A parameter specifying whether the constant zeroth order effects should be added to the
        ALE first order effects.
    ax
        A `matplotlib` axes object or a numpy array of `matplotlib` axes to plot on.
    line_kw
        Keyword arguments passed to the `plt.plot` function.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    An array of matplotlib axes with the resulting ALE plots.

    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # line_kw and fig_kw values
    default_line_kw = {'markersize': 3, 'marker': 'o', 'label': None}
    if line_kw is None:
        line_kw = {}
    line_kw = {**default_line_kw, **line_kw}

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}
    fig_kw = {**default_fig_kw, **fig_kw}

    if features == 'all':
        features = range(0, len(exp.feature_names))
    else:
        for ix, f in enumerate(features):
            if isinstance(f, str):
                try:
                    f = np.argwhere(exp.feature_names == f).item()
                except ValueError:
                    raise ValueError(f"Feature name {f} does not exist.")
            features[ix] = f
    n_features = len(features)

    if targets == 'all':
        targets = range(0, len(exp.target_names))
    else:
        for ix, t in enumerate(targets):
            if isinstance(t, str):
                try:
                    t = np.argwhere(exp.target_names == t).item()
                except ValueError:
                    raise ValueError(f"Target name {t} does not exist.")
            targets[ix] = t

    # make axes
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_features != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_features)
        n_rows = math.ceil(n_features / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        # gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_subplotspec())
        gs = GridSpec(n_rows, n_cols)
        for i, spec in zip(range(n_features), gs):
            # determine which y-axes should be shared
            if sharey == 'all':
                cond = i != 0
            elif sharey == 'row':
                cond = i % n_cols != 0
            else:
                cond = False

            if cond:
                axes_ravel[i] = fig.add_subplot(spec, sharey=axes_ravel[i - 1])
                continue
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_features:
            raise ValueError(f"Expected ax to have {n_features} axes, got {ax.size}")
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    # make plots
    for ix, feature, ax_ravel in \
            zip(count(), features, axes_ravel):
        _ = _plot_one_ale_num(exp=exp,
                              feature=feature,
                              targets=targets,
                              constant=constant,
                              ax=ax_ravel,
                              legend=not ix,  # only one legend
                              line_kw=line_kw)

    # if explicit labels passed, handle the legend here as the axis passed might be repeated
    if line_kw['label'] is not None:
        axes_ravel[0].legend()

    fig.set(**fig_kw)
    # TODO: should we return just axes or ax + axes
    return axes


@no_type_check
def _plot_one_ale_num(exp: Explanation,
                      feature: int,
                      targets: List[int],
                      constant: bool = False,
                      ax: 'plt.Axes' = None,
                      legend: bool = True,
                      line_kw: dict = None) -> 'plt.Axes':
    """
    Plots the ALE of exactly one feature on one axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    # add zero baseline
    ax.axhline(0, color='grey')

    lines = ax.plot(
        exp.feature_values[feature],
        exp.ale_values[feature][:, targets] + constant * exp.constant_value,
        **line_kw
    )

    # add decile markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.feature_deciles[feature][1:], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.feature_names[feature])
    ax.set_ylabel('ALE')

    if legend:
        # if no explicit labels passed, just use target names
        if line_kw['label'] is None:
            ax.legend(lines, exp.target_names[targets])

    return ax
