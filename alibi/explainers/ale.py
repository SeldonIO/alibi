import copy
import math
import numpy as np
import pandas as pd
from itertools import count
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, Dict, TYPE_CHECKING, no_type_check

import sys
import logging

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ALE, DEFAULT_DATA_ALE

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ALE(Explainer):

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None,
                 check_feature_resolution: bool = True,
                 low_resolution_threshold: int = 10,
                 extrapolate_constant: bool = True,
                 extrapolate_constant_perc: float = 10.,
                 extrapolate_constant_min: float = 0.1) -> None:
        """
        Accumulated Local Effects for tabular datasets. Current implementation supports first order
        feature effects of numerical features.

        Parameters
        ----------
        predictor
            A callable that takes in an `N x F` array as input and outputs an `N x T` array (`N` - number of
            data points, `F` - number of features, `T` - number of outputs/targets (e.g. 1 for single output
            regression, >=2 for classification)).
        feature_names
            A list of feature names used for displaying results.
        target_names
            A list of target/output names used for displaying results.
        check_feature_resolution
            If ``True``, the number of unique values is calculated for each feature and if it is less than
            `low_resolution_threshold` then the feature values are used for grid-points instead of quantiles.
            This may increase the runtime of the algorithm for large datasets. Only used for features without custom
            grid-points specified in :py:meth:`alibi.explainers.ale.ALE.explain`.
        low_resolution_threshold
            If a feature has at most this many unique values, these are used as the grid points instead of
            quantiles. This is to avoid situations when the quantile algorithm returns quantiles between discrete
            values which can result in jumps in the ALE plot obscuring the true effect. Only used if
            `check_feature_resolution` is ``True`` and for features without custom grid-points specified in
            :py:meth:`alibi.explainers.ale.ALE.explain`.
        extrapolate_constant
            If a feature is constant, only one quantile exists where all the data points lie. In this case the
            ALE value at that point is zero, however this may be misleading if the feature does have an effect on
            the model. If this parameter is set to ``True``, the ALE values are calculated on an interval surrounding
            the constant value. The interval length is controlled by the `extrapolate_constant_perc` and
            `extrapolate_constant_min` arguments.
        extrapolate_constant_perc
            Percentage by which to extrapolate a constant feature value to create an interval for ALE calculation.
            If `q` is the constant feature value, creates an interval
            `[q - q/extrapolate_constant_perc, q + q/extrapolate_constant_perc]` for which ALE is calculated.
            Only relevant if `extrapolate_constant` is set to ``True``.
        extrapolate_constant_min
            Controls the minimum extrapolation length for constant features. An interval constructed for constant
            features is guaranteed to be `2 x extrapolate_constant_min` wide centered on the feature value. This allows
            for capturing model behaviour around constant features which have small value so that
            `extrapolate_constant_perc` is not so helpful.
            Only relevant if `extrapolate_constant` is set to ``True``.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ALE))

        self.predictor = predictor
        self.feature_names = feature_names
        self.target_names = target_names
        self.check_feature_resolution = check_feature_resolution
        self.low_resolution_threshold = low_resolution_threshold
        self.extrapolate_constant = extrapolate_constant
        self.extrapolate_constant_perc = extrapolate_constant_perc
        self.extrapolate_constant_min = extrapolate_constant_min

        self.meta['params'].update(check_feature_resolution=check_feature_resolution,
                                   low_resolution_threshold=low_resolution_threshold,
                                   extrapolate_constant=extrapolate_constant,
                                   extrapolate_constant_perc=extrapolate_constant_perc,
                                   extrapolate_constant_min=extrapolate_constant_min)

    def explain(self,
                X: np.ndarray,
                features: Optional[List[int]] = None,
                min_bin_points: int = 4,
                grid_points: Optional[Dict[int, np.ndarray]] = None) -> Explanation:
        """
        Calculate the ALE curves for each feature with respect to the dataset `X`.

        Parameters
        ----------
        X
            An `N x F` tabular dataset used to calculate the ALE curves. This is typically the training dataset
            or a representative sample.
        features:
            Features for which to calculate ALE.
        min_bin_points
            Minimum number of points each discretized interval should contain to ensure more precise
            ALE estimation.
        grid_points
            Custom grid points. Must be a `dict` where the keys are features indices and the values are
            monotonically increasing `numpy` arrays defining the grid points for each feature.
            See the :ref:`Notes<Notes ALE explain>` section for the default behavior when potential edge-cases arise
            when using grid-points. If no grid points are specified (i.e. the feature is missing from the `grid_points`
            dictionary), deciles discretization is used instead.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated ALE curves.
            See usage at `ALE examples`_ for details.

            .. _ALE examples:
                https://docs.seldon.io/projects/alibi/en/latest/methods/ALE.html

        Notes
        -----
        .. _Notes ALE explain:

        Consider `f` to be a feature of interest. We denote possible feature values of `f` by `X` (i.e. the values
        from the dataset column corresponding to feature `f`), by `O` a user-specified grid-point value, and by
        `(X|O)` an overlap between a grid-point and a feature value. We can encounter the following edge-cases:

         - Grid points outside the feature range. Consider the following example: `O O O X X O X O X O O`, \
        where 3 grid-points are smaller than the minimum value in `f`, and 2 grid-points are larger than the maximum \
        value in `f`. Grid-points outside the feature value range are clipped between the minimum and maximum \
        values of `f`. The grid-points considered will be: `(O|X) X O X O (X|O)`.

         - Grid points that do not cover the entire feature range. Consider the following example: \
        `X X O X X O X O X X X X X`. Two auxiliary grid-points are added which correspond the value of the minimum \
        and maximum value of feature `f`. The grid-points considered will be: `(O|X) X O X X O X O X X X X (X|O)`.

         - Grid points that do not contain any values in between. Consider the following example: \
        `(O|X) X X O O O X O X O O (X|O)`. The intervals which do not contain any feature values are removed/merged. \
        The grid-points considered will be: `(O|X) X X O X O X O (X|O)`.

        """
        self.meta['params'].update(min_bin_points=min_bin_points)

        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional')
        n_features = X.shape[1]

        # set feature and target names, this is done here as we don't know n_features at init time
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]
        if self.target_names is None:
            pred = np.atleast_2d(self.predictor(X[0].reshape(1, -1)))
            n_targets = pred.shape[1]
            self.target_names = [f'c_{i}' for i in range(n_targets)]
        self.feature_names = np.array(self.feature_names)  # type: ignore
        self.target_names = np.array(self.target_names)  # type: ignore

        # only calculate ALE for the specified features and return the explanation for this subset
        if features:
            feature_names = self.feature_names[features]  # type: ignore
        else:
            feature_names = self.feature_names
            features = list(range(n_features))

        feature_values = []
        ale_values = []
        ale0 = []
        feature_deciles = []

        if grid_points is None:
            grid_points = {}

        # TODO: use joblib to parallelize?
        for feature in features:

            # Getting custom grid values. If the grid for a feature is not specified, `feature_grid_points = None`.
            feature_grid_points = grid_points.get(feature)
            fvals, ale, a0 = ale_num(
                self.predictor,
                X=X,
                feature=feature,
                feature_grid_points=feature_grid_points,
                min_bin_points=min_bin_points,
                check_feature_resolution=self.check_feature_resolution,
                low_resolution_threshold=self.low_resolution_threshold,
                extrapolate_constant=self.extrapolate_constant,
                extrapolate_constant_perc=self.extrapolate_constant_perc,
                extrapolate_constant_min=self.extrapolate_constant_min,
            )
            deciles = get_quantiles(X[:, feature], num_quantiles=11)

            feature_values.append(fvals)
            ale_values.append(ale)
            ale0.append(a0)
            feature_deciles.append(deciles)

        constant_value = self.predictor(X).mean()
        # TODO: an ALE plot ideally requires a rugplot to gauge density of instances in the feature space.
        # I've replaced this with feature deciles which is coarser but has constant space complexity
        # as opposed to a rugplot. Alternatively, could consider subsampling to produce a rug with some
        # maximum number of points.
        return self._build_explanation(
            ale_values=ale_values,
            ale0=ale0,
            constant_value=constant_value,
            feature_values=feature_values,
            feature_deciles=feature_deciles,
            feature_names=feature_names
        )

    def _build_explanation(self,
                           ale_values: List[np.ndarray],
                           ale0: List[np.ndarray],
                           constant_value: float,
                           feature_values: List[np.ndarray],
                           feature_deciles: List[np.ndarray],
                           feature_names: np.ndarray) -> Explanation:
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
            feature_names=feature_names,
            target_names=self.target_names,
            feature_deciles=feature_deciles
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def reset_predictor(self, predictor: Callable) -> None:
        """
        Resets the predictor function.

        Parameters
        ----------
        predictor
            New predictor function.
        """
        self.predictor = predictor


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
    quantiles = np.percentile(values, percentiles, axis=0, interpolation=interpolation)  # type: ignore[call-overload]
    return quantiles


def bisect_fun(fun: Callable, target: float, lo: int, hi: int) -> int:
    """
    Bisection algorithm for function evaluation with integer support.

    Assumes the function is non-decreasing on the interval `[lo, hi]`.
    Return an integer value v such that for all `x<v, fun(x)<target` and for all `x>=v, fun(x)>=target`.
    This is equivalent to the library function `bisect.bisect_left` but for functions defined on integers.

    Parameters
    ----------
    fun
        A function defined on integers in the range `[lo, hi]` and returning floats.
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
    Calculates whether the partition into bins induced by `n` quantiles
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
        feature_grid_points: Optional[np.ndarray] = None,
        min_bin_points: int = 4,
        check_feature_resolution: bool = True,
        low_resolution_threshold: int = 10,
        extrapolate_constant: bool = True,
        extrapolate_constant_perc: float = 10.,
        extrapolate_constant_min: float = 0.1) -> Tuple[np.ndarray, ...]:
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
    feature_grid_points
        Custom grid points. An `numpy` array defining the grid points for the given features.
    min_bin_points
        Minimum number of points each discretized interval should contain to ensure more precise
        ALE estimation.
    check_feature_resolution
        Refer to :class:`ALE` documentation.
    low_resolution_threshold
        Refer to :class:`ALE` documentation.
    extrapolate_constant
        Refer to :class:`ALE` documentation.
    extrapolate_constant_perc
        Refer to :class:`ALE` documentation.
    extrapolate_constant_min
        Refer to :class:`ALE` documentation.

    Returns
    -------
    fvals
        Array of quantiles or custom grid-points of the input values.
    ale
        ALE values for each feature at each of the points in `fvals`.
    ale0
        The constant offset used to center the ALE curves.

    """
    if feature_grid_points is None:
        if check_feature_resolution:
            uniques = np.unique(X[:, feature])
            if len(uniques) <= low_resolution_threshold:
                fvals = uniques
            else:
                fvals, _ = adaptive_grid(X[:, feature], min_bin_points)
        else:
            fvals, _ = adaptive_grid(X[:, feature], min_bin_points)
    else:
        # set q to custom grid for feature
        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        fvals = np.sort(feature_grid_points)

        if min_val > fvals[0]:
            logger.warning(f'Feature {feature} grid-points contain lower values than the minimum feature value. '
                           'Automatically lower bound clipping the grid-points values.')

        if max_val < fvals[-1]:
            logger.warning(f'Feature {feature} grid-points contain larger values than the maximum feature value. '
                           'Automatically upper bound clipping the grid-points values.')

        # clip the values and remove duplicates
        fvals = np.unique(np.clip(fvals, a_min=min_val, a_max=max_val))

        # add min feature value and maybe log a warning
        if fvals[0] > min_val:
            fvals = np.append(min_val, fvals)
            logger.warning(f'Feature {feature} grid-points does not cover the lower feature values. '
                           'Automatically adding the minimum feature values to the grid-points.')

        # add max feature value and maybe log a warning
        if fvals[-1] < max_val:
            fvals = np.append(fvals, max_val)
            logger.warning(f'Feature {feature} grid-points does not cover the larger feature values. '
                           'Automatically adding the maximum feature value to the grid points.')

        # check how many feature values are in each bin
        indices = np.searchsorted(fvals, X[:, feature], side="left")
        interval_n = np.bincount(indices)  # number of points in each interval

        if np.any(interval_n == 0):
            fvals = np.delete(fvals, np.where(interval_n == 0)[0])
            logger.warning(f'Some bins of feature {feature} defined by the grid-points do not contain '
                           'any feature values. Automatically merging consecutive bins to ensure that '
                           'each bin contains at least one value.')

    # if the feature is constant, calculate the ALE on a small interval surrounding the feature value
    if len(fvals) == 1:
        if extrapolate_constant:
            delta = max(fvals * extrapolate_constant_perc / 100, extrapolate_constant_min)
            fvals = np.hstack((fvals - delta, fvals + delta))
        else:
            # ALE is 0 at a constant feature value
            return fvals, np.array([[0.]]), np.array([0.])

    # find which interval each observation falls into
    indices = np.searchsorted(fvals, X[:, feature], side="left")
    indices[indices == 0] = 1  # put the smallest data point in the first interval
    interval_n = np.bincount(indices)  # number of points in each interval

    # predictions for the upper and lower ranges of intervals
    z_low = X.copy()
    z_high = X.copy()
    z_low[:, feature] = fvals[indices - 1]
    z_high[:, feature] = fvals[indices]
    p_low = predictor(z_low)
    p_high = predictor(z_high)

    # finite differences
    p_deltas = p_high - p_low

    # make a dataframe for averaging over intervals
    concat = np.column_stack((p_deltas, indices))
    df = pd.DataFrame(concat)
    avg_p_deltas = df.groupby(df.shape[1] - 1).mean().values  # groupby indices

    # accumulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

    # pre-pend 0 for the left-most point
    zeros = np.zeros((1, accum_p_deltas.shape[1]))
    accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

    # mean effect, R's `ALEPlot` and `iml` version (approximation per interval)
    ale0 = (0.5 * (accum_p_deltas[:-1, :] + accum_p_deltas[1:, :]) * interval_n[1:, np.newaxis]).sum(axis=0)
    ale0 = ale0 / interval_n.sum()

    # crude approximation (assume data points on interval endpoints)
    # ale0 = accum_p_deltas.mean(axis=0)

    # exact marginalisation
    # exact_ale = accum_p_deltas[indices - 1] + ((X[:, feature] - q[indices])) / (q[indices] - q[indices - 1]) * (
    #            accum_p_deltas[indices] - accum_p_deltas[indices - 1])
    # ale0 = exact_ale.mean()

    # center
    ale = accum_p_deltas - ale0

    return fvals, ale, ale0


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
        An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.ale.ALE.explain` method.
    features
        A list of features for which to plot the ALE curves or ``'all'`` for all features.
        Can be a mix of integers denoting feature index or strings denoting entries in
        `exp.feature_names`. Defaults to ``'all'``.
    targets
        A list of targets for which to plot the ALE curves or ``'all'`` for all targets.
        Can be a mix of integers denoting target index or strings denoting entries in
        `exp.target_names`. Defaults to ``'all'``.
    n_cols
        Number of columns to organize the resulting plot into.
    sharey
        A parameter specifying whether the y-axis of the ALE curves should be on the same scale
        for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``.
    constant
        A parameter specifying whether the constant zeroth order effects should be added to the
        ALE first order effects.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    line_kw
        Keyword arguments passed to the `plt.plot` function.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    An array of `matplotlib` axes with the resulting ALE plots.

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
