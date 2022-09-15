import copy
import logging
import math
import numbers
import sys
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from alibi.api.defaults import DEFAULT_DATA_PDVARIANCE, DEFAULT_META_PDVARIANCE
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.partial_dependence import (Kind, PartialDependence,
                                                 TreePartialDependence)
from alibi.explainers.similarity.grad import get_options_string
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class Method(str, Enum):
    """ Enumeration of supported methods. """
    IMPORTANCE = 'importance'
    INTERACTION = 'interaction'


class PartialDependenceVariance(Explainer):
    """ Implementation of the variance partial dependence feature importance for tabular datasets. Supports
     black-box models and the following `sklearn` tree-based models: `GradientBoostingClassifier`,
     `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`,
     `HistGradientBoostingRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`."""

    def __init__(self,
                 predictor: Union[BaseEstimator, Callable[[np.ndarray], np.ndarray]],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize black-box model implementation for the variance partial dependence feature importance.

        Parameters
        ----------
        predictor
             A `sklearn` estimator or a prediction function which receives as input a `numpy` array of size `N x F`
            and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input
            instances, `F` is the number of features and `T` is the number of targets.
        feature_names
            A list of feature names used for displaying results.E
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature. Necessary to
            identify the categorical features in the dataset. An example for `categorical_names` would be::

                category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]}

        target_names
            A list of target/output names used for displaying results.
        verbose
            Whether to print the progress of the explainer.

        Notes
        -----
        The length of the `target_names` should match the number of columns returned by a call to the `predictor`.
        For example, in the case of a binary classifier, if the predictor outputs a decision score (i.e. uses
        the `decision_function` method) which returns one column, then the length of the `target_names` should be one.
        On the other hand, if the predictor outputs a prediction probability (i.e. uses the `predict_proba` method)
        which returns two columns (one for the negative class and one for the positive class), then the length of
        the `target_names` should be two.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PDVARIANCE))

        # initialize the pd explainer
        PartialDependenceClass = TreePartialDependence if isinstance(predictor, BaseEstimator) else PartialDependence
        self.pd_explainer = PartialDependenceClass(predictor=predictor,
                                                   feature_names=feature_names,
                                                   categorical_names=categorical_names,
                                                   target_names=target_names,
                                                   verbose=verbose)

    def explain(self,
                X: np.ndarray,
                features: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
                method: Literal['importance', 'interaction'] = 'importance',
                percentiles: Tuple[float, float] = (0., 1.),
                grid_resolution: int = 100,
                grid_points: Optional[Dict[int, Union[List, np.ndarray]]] = None) -> Explanation:
        """
        Calculates the variance partial dependence feature importance for each feature with respect to the all targets
        and the reference dataset `X`.

        Parameters
        ----------
        X
            A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the
            training dataset or a representative sample.
        features
            A list of features for which to compute the feature importance or a list of feature pairs
            for which to compute the feature interaction. Some example of `features` would be: ``[0, 1, 3]``,
            ``[(0, 1), (0, 3), (1, 3)]``, where ``0``,``1``, and ``3`` correspond to the columns 0, 1, and 3 in `X`.
            If not provided, the feature importance or the feature interaction will be computed for every
            feature or for every combination of features, depending on the parameter `mode`.
        method
            Flag to specify whether to compute the feature importance or the feature interaction of the elements in
            provided in `features`. Supported values: ``'importance'`` | ``'interaction'``.
        percentiles
            Lower and upper percentiles used to limit the feature values to potentially remove outliers from
            low-density regions. Note that for features with not many data points with large/low values, the
            PD estimates are less reliable in those extreme regions. The values must be in [0, 1]. Only used
            with `grid_resolution`.
        grid_resolution
            Number of equidistant points to split the range of each target feature. Only applies if the number of
            unique values of a target feature in the reference dataset `X` is greater than the `grid_resolution` value.
            For example, consider a case where a feature can take the following values:
            ``[0.1, 0.3, 0.35, 0.351, 0.4, 0.41, 0.44, ..., 0.5, 0.54, 0.56, 0.6, 0.65, 0.7, 0.9]``, and we are not
            interested in evaluating the marginal effect at every single point as it can become computationally costly
            (assume hundreds/thousands of points) without providing any additional information for nearby points
            (e.g., 0.35 and 351). By setting ``grid_resolution=5``, the marginal effect is computed for the values
            ``[0.1, 0.3, 0.5, 0.7, 0.9]`` instead, which is less computationally demanding and can provide similar
            insights regarding the model's behaviour. Note that the extreme values of the grid can be controlled
            using the `percentiles` argument.
        grid_points
            Custom grid points. Must be a `dict` where the keys are the target features indices and the values are
            monotonically increasing arrays defining the grid points for a numerical feature, and a subset of
            categorical feature values for a categorical feature. If the `grid_points` are not specified,
            then the grid will be constructed based on the unique target feature values available in the
            dataset `X`, or based on the `grid_resolution` and `percentiles` (check `grid_resolution` to see when
            it applies). For categorical features, the corresponding value in the `grid_points` can be
            specified either as array of strings or array of integers corresponding the label encodings.
            Note that the label encoding must match the ordering of the values provided in the `categorical_names`.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated partial dependence
            curves and feature importance/interaction. See usage at `Partial dependence variance examples`_ for details

            .. _Partial dependence variance examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html
        """
        if method not in Method.__members__.values():
            raise ValueError(f"Unknown mode. Received ``method={method}``. "
                             f"Accepted `method` names are: {get_options_string(Method)}")

        # get number of features
        n_features = X.shape[1]

        # construct features if not provided based on the method
        if features is None:
            features = list(range(n_features))

            if method == Method.INTERACTION:
                features = list(combinations(features, 2))

        # compute partial dependence functions
        params = {
            'X': X,
            'features': features,
            'percentiles': percentiles,
            'grid_resolution': grid_resolution,
            'grid_points': grid_points
        }

        if not isinstance(self.pd_explainer.predictor, BaseEstimator):
            params.update({'kind': Kind.AVERAGE.value})  # type: ignore[dict-item]

        # compute partial dependence for each feature
        pd_explanation = self.pd_explainer.explain(**params)  # type: ignore[arg-type]

        if method == Method.IMPORTANCE:
            if not all([isinstance(f, numbers.Integral) for f in features]):
                raise ValueError(f"For ``method='{Method.IMPORTANCE.value}'`` all features must be integers.")
            # compute feature importance
            buffers = self._compute_feature_importance(pd_explanation=pd_explanation, features=features)
        else:
            if not all([isinstance(fs, tuple) and len(fs) for fs in features]):
                raise ValueError(f"For ``method='{Method.INTERACTION.value}'`` all features must be "
                                 f"tuples of length 2.")
            # compute feature interaction
            buffers = self._compute_feature_interaction(pd_explanation=pd_explanation,
                                                        features=features)  # type: ignore[arg-type]

        # update meta and return built explanation
        self.meta['params'].update(self.pd_explainer.meta['params'])
        self.meta['params'].update({'method': method})
        return self._build_explanation(buffers=buffers)

    def _compute_pd_variance(self, features: List[int], pd_values: List[np.ndarray]) -> np.ndarray:
        """
        Computes the PD variance along the final axis for all the features.

        Parameters
        ----------
        features
            See :py:meth:`alibi.explainers.pd_variance.PartialDependenceVariance.explain`.
        pd_values
            List of length `F` containing the PD values for each feature in `features`. Each PD value is an array
            with the shape `T x N1 ... x Nk` where `T` is the number of targets and `Ni` is the number of feature
            values along the `i` axis.

        Returns
        -------
        An array of size `F x T x N1 x ... N(k-1)`, where `F` is the number of explained features, `T` is the number
        of targets, and `Ni` is the number of feature values along the `i` axis.
        """
        feature_variance = []

        for pdv, f in zip(pd_values, features):
            # `pdv` is a tensor of size `T x N1 x ... Nk`, where `T` is the number
            # of targets and `Ni` is the number of feature values along the axis `i`
            if f in self.pd_explainer.categorical_names:  # type: ignore[operator]
                ft_var = PartialDependenceVariance._compute_pd_variance_cat(pdv)
            else:
                ft_var = PartialDependenceVariance._compute_pd_variance_num(pdv)
            # add extra dimension for later concatenation along the axis 0
            feature_variance.append(ft_var[None, ...])  # type: ignore[index]

        # stack the feature importance such that the array has the shape `F x T x N1 ... N(k-1)`
        return np.concatenate(feature_variance, axis=0)

    @staticmethod
    def _compute_pd_variance_num(pd_values: np.ndarray) -> np.ndarray:
        """
        Computes the PD variance along the final axis for a numerical feature.

        Parameters
        ----------
        pd_values
            Array of partial dependence values for a numerical features of size `T x N1 x ... x Nk`, where `T` is the
            number of targets and `Ni` is the number of feature values along the axis `i`.

        Returns
        -------
        PD variance along the final axis for a numerical feature.
        """
        return np.std(pd_values, axis=-1, ddof=1, keepdims=False)

    @staticmethod
    def _compute_pd_variance_cat(pd_values: np.ndarray) -> np.ndarray:
        """
        Computes the PD variance along the final axis for a categorical feature.

        Parameters
        ----------
        pd_values
            Array of partial dependence values for a categorical feature of size `T x N1 x ... x Nk`, where `T` is the
            number of targets and `Ni` is the number of feature values along the axis `i`.

        Returns
        -------
        PD variance along the final axis for a categorical feature.
        """
        return (np.max(pd_values, axis=-1, keepdims=False) - np.min(pd_values, axis=-1, keepdims=False)) / 4

    def _compute_feature_importance(self,
                                    features: List[int],
                                    pd_explanation: Explanation) -> Dict[str, Any]:
        """
        Computes the feature importance.

        Parameters
        ----------
        features
            List of features to compute the importance for.
        pd_explanation
            Partial dependence explanation object.

        Returns
        -------
        Dictionary with all the keys necessary to build the explanation.
        """
        return {
            'feature_deciles': pd_explanation.data['feature_deciles'],
            'pd_values': pd_explanation.data['pd_values'],
            'feature_values': pd_explanation.data['feature_values'],
            'feature_names': pd_explanation.data['feature_names'],
            'feature_importance': self._compute_pd_variance(features=features,
                                                            pd_values=pd_explanation.data['pd_values']).T,
        }

    def _compute_feature_interaction(self,
                                     features: List[Tuple[int, int]],
                                     pd_explanation: Explanation) -> Dict[str, Any]:
        """
        Computes the feature interactions.

        Parameters
        ----------
        features
            List of feature pairs to compute the interaction for.
        pd_explanation
            Partial dependence explanation object.

        Returns
        -------
        Dictionary with all the keys necessary to build the explanation.
        """
        buffers: Dict[str, Any] = {
            'feature_interaction': [],
            'feature_deciles': [],
            'pd_values': [],
            'feature_values': [],
            'feature_names': []
        }

        for i in range(len(features)):
            # unpack explanation
            feature_deciles = pd_explanation.data['feature_deciles'][i]
            pd_values = pd_explanation.data['pd_values'][i]
            feature_values = pd_explanation.data['feature_values'][i]
            feature_names = pd_explanation.data['feature_names'][i]

            # append data for the 2-way pdp
            buffers['feature_deciles'].append(feature_deciles)
            buffers['pd_values'].append(pd_values)
            buffers['feature_values'].append(feature_values)
            buffers['feature_names'].append(feature_names)

            # compute variance when keeping f0 value constant and vary f1.
            # Note that we remove the first axis here since we are dealing with only one feature
            feature_interaction = []

            for j in range(2):
                tmp_pd_values = pd_values if j == 0 else pd_values.transpose(0, 2, 1)
                cond_pd_values = self._compute_pd_variance(features=[features[i][1 - j]], pd_values=[tmp_pd_values])[0]
                buffers['feature_deciles'].append(feature_deciles[j])
                buffers['pd_values'].append(cond_pd_values)
                buffers['feature_values'].append(feature_values[j])
                buffers['feature_names'].append(feature_names[j])
                feature_interaction.append(
                    self._compute_pd_variance(features=[features[i][1 - j]], pd_values=[cond_pd_values])[0]
                )

            # compute the feature interaction as the average of the two
            buffers['feature_interaction'].append(np.mean(feature_interaction, axis=0, keepdims=True))

        # transform `feature_interaction` into an array of shape `T x F`, where `T` is the number of targets
        # and `F` is the number of feature pairs.
        buffers['feature_interaction'] = np.concatenate(buffers['feature_interaction'], axis=0).T
        return buffers

    def _build_explanation(self, buffers: dict) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        buffers
            Dictionary with all the data necessary to build the explanation.

        Returns
        -------
        `Explanation` object.
        """
        data = copy.deepcopy(DEFAULT_DATA_PDVARIANCE)
        data.update(feature_deciles=buffers['feature_deciles'],
                    pd_values=buffers['pd_values'],
                    feature_values=buffers['feature_values'],
                    feature_names=buffers['feature_names'])

        if self.meta['params']['method'] == Method.IMPORTANCE:
            data.update(feature_importance=buffers['feature_importance'])
        else:
            data.update(feature_interaction=buffers['feature_interaction'])

        return Explanation(meta=self.meta, data=data)


def _plot_hbar(exp_values: np.ndarray,
               exp_feature_names: List[str],
               exp_target_names: List[str],
               features: Union[List[int], Literal['all']] = 'all',
               targets: Union[List[Union[str, int]], Literal['all']] = 'all',
               n_cols: int = 3,
               sort: bool = True,
               top_k: Optional[int] = None,
               title: str = '',
               ax: Optional[Union['plt.Axes', np.ndarray]] = None,
               bar_kw: Optional[dict] = None,
               fig_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Horizontal bar plot.

    Parameters
    ----------
    exp_values
        Explanation values to be plotted.
    exp_feature_names
        A list of explanation feature names. Used as the y-axis labels.
    exp_target_names
        A list of explanation target names. Determines the number of plots (i.e., one for each target).
    features
        A list of features entries of feature_names` in the
        :py:meth:`alibi.explainers.pd_variance.PartialDependence.explain`
        to plot the partial dependence curves for, or ``'all'`` to plot all the explained feature or tuples of features.
        This includes tuples of features. For example, if ``feature_names = ['temp', 'hum', 'windspeed']``
        and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    targets
        The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers
        denoting target index or strings denoting entries in `exp.meta['params']['target_names']`.
    n_cols
        Number of columns to organize the resulting plot into.
    sort
        Boolean flag whether to sort the values in descending order.
    top_k
        Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed.
    title
        The title of the bar plot.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    bar_kw
        Keyword arguments passed to the `matplotlib.pyplot.barh`_ function.
    fig_kw
        Keyword arguments passed to the `matplotlib.figure.set`_ function.

        .. _matplotlib.pyplot.barh`
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    -------
    `plt.Axes` with the values plot.
    """
    from matplotlib.gridspec import GridSpec

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}

    fig_kw = {**default_fig_kw, **fig_kw}

    if features == 'all':
        feature_indices = list(range(len(exp_feature_names)))
        feature_names = exp_feature_names
    else:
        feature_indices, feature_names = features, []
        for ifeatures in features:
            if ifeatures > len(exp_feature_names):
                raise ValueError(f"The `features` indices must be less than the "
                                 f"``len(feature_names) = {len(exp_feature_names)}``. "
                                 f"Received {ifeatures}.")
            else:
                feature_names.append(exp_feature_names[ifeatures])

    # set target indices
    if targets == 'all':
        target_indices = list(range(len(exp_target_names)))
        target_names = exp_target_names
    else:
        target_indices, target_names = [], []

        for target in targets:
            if isinstance(target, str):
                try:
                    target_idx = exp_target_names.index(target)
                    target_name = target
                except ValueError:
                    raise ValueError(f"Unknown `target` name. Received {target}. "
                                     f"Available values are: {exp_target_names}.")
            else:
                try:
                    target_idx = target
                    target_name = exp_target_names[target]
                except IndexError:
                    raise IndexError(f"Target index out of range. Received {target}. "
                                     f"The number of targets is {len(exp_target_names)}.")
            target_indices.append(target_idx)
            target_names.append(target_name)

    # create axes
    if ax is None:
        fig, ax = plt.subplots()

    # number of targets will correspond to the number of axis
    n_trargets = len(target_names)

    if isinstance(ax, plt.Axes) and n_trargets != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_trargets)
        n_rows = math.ceil(n_trargets / n_cols)
        axes = np.empty((n_rows, n_cols), dtype=np.object)   # type: ignore[attr-defined]
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)

        for i, spec in enumerate(list(gs)[:n_trargets]):
            axes_ravel[i] = fig.add_subplot(spec)
    else:
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_trargets:
            raise ValueError(f"Expected ax to have {n_trargets} axes, got {ax.size}")
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    for i, (target_index, target_name, ax) in enumerate(zip(target_indices, target_names, axes_ravel)):
        width = exp_values[target_index][feature_indices]
        y = np.arange(len(feature_indices))
        y_labels = feature_names

        if sort:
            sorted_indices = np.argsort(width)[::-1]
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError('``top_k`` must be greater than 0.')
                sorted_indices = sorted_indices[:top_k]
                y = y[:top_k]
            width = width[sorted_indices]
            y_labels = [y_labels[j] for j in sorted_indices]

        default_bar_kw = {'align': 'center'}
        bar_kw = default_bar_kw if bar_kw is None else {**default_bar_kw, **bar_kw}
        ax.barh(y=y, width=width, **bar_kw)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(title)
        ax.set_title(target_name)

    fig.set(**fig_kw)
    return axes


def plot_feature_importance(exp: Explanation,
                            features: Union[List[int], Literal['all']] = 'all',
                            targets: Union[List[Union[str, int]], Literal['all']] = 'all',
                            n_cols: int = 3,
                            sort: bool = True,
                            top_k: Optional[int] = None,
                            ax: Optional[Union['plt.Axes', np.ndarray]] = None,
                            bar_kw: Optional[dict] = None,
                            fig_kw: Optional[dict] = None):
    """
    Horizontal bar plot for feature importance.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.pd_varianace.PartialDependenceVariance.explain` method.
    features
        A list of features entries of feature_names` in the
        :py:meth:`alibi.explainers.pd_variance.PartialDependence.explain`
        to plot the partial dependence curves for, or ``'all'`` to plot all the explained feature or tuples of features.
        This includes tuples of features. For example, if ``feature_names = ['temp', 'hum', 'windspeed']``
        and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    targets
        The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers
        denoting target index or strings denoting entries in `exp.meta['params']['target_names']`.
    n_cols
        Number of columns to organize the resulting plot into.
    sort
        Boolean flag whether to sort the values in descending order.
    top_k
        Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    bar_kw
        Keyword arguments passed to the `matplotlib.pyplot.barh`_ function.
    fig_kw
        Keyword arguments passed to the `matplotlib.figure.set`_ function.

        .. _matplotlib.pyplot.barh`
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    -------
    `plt.Axes` with the feature importance plot.
    """
    return _plot_hbar(exp_values=exp.data['feature_importance'],
                      exp_feature_names=exp.data['feature_names'],
                      exp_target_names=exp.meta['params']['target_names'],
                      features=features,
                      targets=targets,
                      n_cols=n_cols,
                      sort=sort,
                      top_k=top_k,
                      title='Feature importance',
                      ax=ax,
                      bar_kw=bar_kw,
                      fig_kw=fig_kw)


def plot_feature_interaction(exp: Explanation,
                             features: Union[List[int], Literal['all']] = 'all',
                             targets: Union[List[Union[str, int]], Literal['all']] = 'all',
                             n_cols: int = 3,
                             sort: bool = True,
                             top_k: Optional[int] = None,
                             ax: Optional[Union['plt.Axes', np.ndarray]] = None,
                             bar_kw: Optional[dict] = None,
                             fig_kw: Optional[dict] = None):

    """
    Horizontal bar plot for feature interaction.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.pd_varianace.PartialDependenceVariance.explain` method.
    features
        A list of features entries of feature_names` in the
        :py:meth:`alibi.explainers.pd_variance.PartialDependence.explain`
        to plot the partial dependence curves for, or ``'all'`` to plot all the explained feature or tuples of features.
        This includes tuples of features. For example, if ``feature_names = ['temp', 'hum', 'windspeed']``
        and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    targets
        The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers
        denoting target index or strings denoting entries in `exp.meta['params']['target_names']`.
    n_cols
        Number of columns to organize the resulting plot into.
    sort
        Boolean flag whether to sort the values in descending order.
    top_k
        Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    bar_kw
        Keyword arguments passed to the `matplotlib.pyplot.barh`_ function.
    fig_kw
        Keyword arguments passed to the `matplotlib.figure.set`_ function.

        .. _matplotlib.pyplot.barh`
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    -------
    `plt.Axes` with the feature interaction plot.
    """
    feature_names = ['({}, {})'.format(*fs) for fs in exp.data['feature_names'] if isinstance(fs, tuple)]
    return _plot_hbar(exp_values=exp.data['feature_interaction'],
                      exp_feature_names=feature_names,
                      exp_target_names=exp.meta['params']['target_names'],
                      features=features,
                      targets=targets,
                      n_cols=n_cols,
                      sort=sort,
                      top_k=top_k,
                      title='Feature interaction',
                      ax=ax,
                      bar_kw=bar_kw,
                      fig_kw=fig_kw)
