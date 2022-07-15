import copy
import math
import numbers
import logging

import matplotlib.pyplot as plt
import numpy as np


from enum import Enum
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal

from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.extmath import cartesian
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import BaseHistGradientBoosting
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.inspection._partial_dependence import (
    _grid_from_X,
    _partial_dependence_brute,
    _partial_dependence_recursion
)

from alibi.explainers.ale import get_quantiles
from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_PD, DEFAULT_DATA_PD

logger = logging.getLogger(__name__)


def get_options_string(enum: Type[Enum]) -> str:
    """Get the enums options seperated by pipe as a string."""
    return f"""'{"' | '".join(enum)}'"""


class ResponseMethod(str, Enum):
    """
    Enumeration of supported response methods.
    """
    AUTO = 'auto'
    PREDICT_PROBA = 'predict_proba'
    DECISION_FUNCTION = 'decision_function'


class Method(str, Enum):
    """
    Enumeration of supported methods.
    """
    AUTO = 'auto'
    RECURSION = 'recursion'
    BRUTE = 'brute'


class Kind(str, Enum):
    """
    Enumeration of supported kinds.
    """
    AVERAGE = 'average'
    INDIVIDUAL = 'individual'
    BOTH = 'both'


class PartialDependence(Explainer):
    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None):
        """
        Partial Dependence for tabular datasets. Supports one feature or two feature interactions.

        Parameters
        ----------
        predictor
            A callable that takes in an `N x F` array as input and outputs an `N x T` array (`N` -number of data
            points, `F` - number of features, `T` - number of outputs/targets.
        feature_names
            A list of feature names used for displaying results.
        target_names
            A list of target/output names used for displaying results.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PD))

        self.predictor = predictor
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.target_names = target_names


    def explain(self,
                X: np.ndarray,
                features_list: List[Union[int, Tuple[int, int]]],
                response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                percentiles: Tuple[float, float] = (0.05, 0.95),
                grid_resolution: int = 100,
                method: Literal['auto', 'recursion', 'brute'] = 'auto',
                kind: Literal['average', 'individual', 'both'] = 'average') -> Explanation:
        """
        Calculate the Partial Dependence for each feature with respect to the given target and the dataset `X`.

        Parameters
        ----------
        X
            An `N x F` tabular dataset used to calculate Partial Dependence curves. This is typically the training
            dataset or a representative sample.
        features_list
            Features for which to calculate the Partial Dependence.
        response_method
            Specifies the prediction function to be used. For classifier it specifies whether to use the
            `predict_proba` or the `decision_function`. For a regressor, the parameter is ignored. If set to `auto`,
            the `predict_proba` is tried first, and if not supported then it reverts to `decision_function`. Note
            that if `method='recursion'`, the that the prediction function always uses `decision_function`.
        percentiles
            Lower and upper percentiles used to create extreme values which can potential remove outliers in low
            density regions. The values must be in [0, 1].
        grid_resolution
            Number of equidistant points to split the range of each target feature.
        method
            The method used to calculate the average predictions

             - `'recursion'` - a faster alternative only supported by some tree-based model. For a classifier, the
             target response is always the decision function and NOT the predicted probabilities. Furthermore, since
             the `'recursion'` method computes implicitly the average of the Individual Conditional Expectation (ICE)
             by design, it is incompatible with ICE and the `kind` parameter must be set to `'average'`. Check the
             `sklearn documentation`_ for a list of supported tree-based classifiers.

            .. _sklearn documentation:
                https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html#sklearn.inspection.partial_dependence

             - `'brute'` - supported for any black-box prediction model, but is more computationally intensive.

             - `'auto'` - uses `'recursion'` if the `predictor` supports it. Otherwise uses the `'brute'` method.
        kind
            If set to `'average'`, then only the Partial Dependence (PD) averaged across all samples from the dataset
            is returned. If set to `individual`, then only the Individual Conditional Expectation (ICE) is returned for
            each individual from the dataset. Otherwise, if set to `'both'`, then both the PD and the ICE are returned.
            Note that for the faster `method='recursion'` option the only compatible parameter value is
            `kind='average'`. To plot the ICE, consider using the more computation intensive `method='brute'`.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated Partial Dependece
            curves. See usage at `Partial Dependence examples`_ for details

            .. _Partial Dependence examples:
                https://docs.seldon.io/projects/alibi/en/latest/methods/PartialDependence.html
        """
        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional.')
        n_features = X.shape[1]

        # set the features_names when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # set categorical_names when the user did not provide the category mapping
        if self.categorical_names is None:
            self.categorical_names = {}

        self.feature_names = np.array(self.feature_names)
        self.categorical_names = {k: np.array(v) for k, v in self.categorical_names.items()}

        # parameters sanity checks
        self._features_sanity_checks(features=features_list)
        response_method, method, kind = self._params_sanity_checks(estimator=self.predictor,
                                                                   response_method=response_method,
                                                                   method=method,
                                                                   kind=kind)

        # compute partial dependencies for every features.
        # TODO: implement parallel version
        pds = []
        feature_names = []

        for features in features_list:
            if isinstance(features, Tuple):
                feature_names.append(tuple([self.feature_names[f] for f in features]))
            else:
                feature_names.append(self.feature_names[features])

            # compute partial dependence
            pds.append(
                self._partial_dependence(
                            estimator=self.predictor,
                            X=X,
                            features=features,
                            response_method=response_method,
                            percentiles=percentiles,
                            grid_resolution=grid_resolution,
                            method=method,
                            kind=kind
                )
            )

        # set the target_names when the user did not provide the target names
        # we do it here to avoid checking model's type, prediction function etc
        if self.target_names is None:
            key = Kind.AVERAGE if kind in [Kind.AVERAGE, Kind.BOTH] else Kind.INDIVIDUAL
            n_targets = pds[features_list[0]][key].shape[0]
            self.target_names = [f'c_{i}' for i in range(n_targets)]

        return self._build_explanation(response_method=response_method,
                                       method=method,
                                       kind=kind,
                                       feature_names=feature_names,
                                       pds=pds)

    def _features_sanity_checks(self, features: List[Union[int, Tuple[int, int]]]):
        """ Features sanity checks. """
        def check_feature(f):
            if not isinstance(f, numbers.Integral):
                raise ValueError(f'All feature entries must be integers. Got a feature value of {type(f)} type.')
            if f >= len(self.feature_names):
                raise ValueError(f'All feature entries must be less than len(feature_names)={len(self.feature_names)}. '
                                 f'Got a feature value of {f}.')
            if f < 0:
                raise ValueError(f'All features entries must be greater or equal to 0. Got a feature value of {f}.')

        for f in features:
            if isinstance(f, Tuple):
                if len(f) > 2:
                    raise ValueError(f'Current implementation of the Partial Dependence supports only up to '
                                     f'two features at a time. Received {len(f)} features with the values {f}.')
                check_feature(f[0])
                check_feature(f[1])
            else:
                check_feature(f)

    def _params_sanity_checks(self,
                              estimator: Callable[[np.ndarray], np.ndarray],
                              response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                              method: Literal['auto', 'recursion', 'brute'] = 'auto',
                              kind: Literal['average', 'individual', 'both'] = 'average'):
        """
        Parameters sanity checks. Most of the code is borrowed from:
        https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/inspection/_partial_dependence.py
        """
        check_is_fitted(estimator)

        if not (is_classifier(estimator) or is_regressor(estimator)):
            raise ValueError('The predictor must be a fitted regressor or a fitted classifier.')

        if is_classifier(estimator) and isinstance(estimator.classes_[0], np.ndarray):
            raise ValueError('Multiclass-multioutput predictors are not supported.')

        if response_method not in ResponseMethod.__members__.values():
            raise ValueError(f"response_method='{response_method}' is invalid. Accepted response_method "
                             f"names are {get_options_string(ResponseMethod)}.")

        if is_regressor(estimator) and response_method != ResponseMethod.AUTO:
            raise ValueError(f"The response_method parameter is ignored for regressor and is "
                             f"automatically set to '{ResponseMethod.AUTO.value}'.")

        if method not in Method.__members__.values():
            raise ValueError(f"method='{method}' is invalid. Accepted method names are {get_options_string(Method)}.")

        if kind != Kind.AVERAGE:
            if method == Method.RECURSION:
                raise ValueError(f"The '{Method.RECURSION.value}' method only applies when kind='average'.")
            method = Method.BRUTE

        if method == Method.AUTO:
            if isinstance(estimator, BaseGradientBoosting) and estimator.init is None:
                method = Method.RECURSION
            elif isinstance(estimator, (BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor)):
                method = Method.RECURSION
            else:
                method = Method.BRUTE

        if method == Method.RECURSION:
            if not isinstance(estimator, (BaseGradientBoosting, BaseHistGradientBoosting, DecisionTreeRegressor,
                                          RandomForestRegressor)):
                supported_classes_recursion = (
                    "GradientBoostingClassifier",
                    "GradientBoostingRegressor",
                    "HistGradientBoostingClassifier",
                    "HistGradientBoostingRegressor",
                    "HistGradientBoostingRegressor",
                    "DecisionTreeRegressor",
                    "RandomForestRegressor",
                )
                raise ValueError(f"Only the following estimators support the 'recursion' "
                                 f"method: {supported_classes_recursion}. Try using method='{Method.BRUTE.value}'.")

            if response_method == ResponseMethod.AUTO:
                response_method = ResponseMethod.DECISION_FUNCTION

            if response_method != ResponseMethod.DECISION_FUNCTION:
                raise ValueError(f"With the '{method.RECURSION.value}' method, the response_method must be "
                                 f"'{response_method.DECISION_FUNCTION.value}'. Got {response_method}.")

        return response_method, method, kind

    def _partial_dependence(self,
                            estimator: Callable[[np.ndarray], np.ndarray],
                            X: np.ndarray,
                            features: Union[int, Tuple[int, int]],
                            response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                            percentiles: Tuple[float, float] = (0., 1.),
                            grid_resolution: int = 100,
                            method: Literal['auto', 'recursion', 'brute'] = 'auto',
                            kind: Literal['average', 'individual', 'both'] = 'average'):

        if isinstance(features, numbers.Integral):
            features = (features, )

        deciles, grid, values, features_indices = [], [], [], []
        for f in features:
            f_indices = np.asarray(_get_column_indices(X, f), dtype=np.int32, order='C').ravel()
            X_f = _safe_indexing(X, f_indices, axis=1)

            # get deciles
            deciles_f = get_quantiles(X_f, num_quantiles=11) if self._is_numerical(f) else None

            # construct grid for feature f
            grid_f, values_f = _grid_from_X(X_f,
                                            percentiles=percentiles,
                                            grid_resolution=grid_resolution if self._is_numerical(f) else np.inf)

            features_indices.append(f_indices)
            deciles.append(deciles_f)
            grid.append(grid_f)
            values += values_f

        # covers also the case of a single feature, just to ensure it has the right shape
        features_indices = np.concatenate(features_indices, axis=0)
        grid = cartesian(tuple([g.reshape(-1) for g in grid]))

        if method == "brute":
            averaged_predictions, predictions = _partial_dependence_brute(
                estimator, grid, features_indices, X, response_method
            )

            # reshape predictions to (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
            predictions = predictions.reshape(
                -1, X.shape[0], *[val.shape[0] for val in values]
            )
        else:
            averaged_predictions = _partial_dependence_recursion(
                estimator, grid, features_indices
            )

            # reshape averaged_predictions to (n_outputs, n_values_feature_0, n_values_feature_1, ...)
        averaged_predictions = averaged_predictions.reshape(
            -1, *[val.shape[0] for val in values]
        )

        pd = {
            'values': values if len(values) > 1 else values[0],
            'deciles': deciles if len(deciles) > 1 else deciles[0],
        }
        if kind == Kind.AVERAGE:
            pd.update({'average': averaged_predictions})
        elif kind == Kind.INDIVIDUAL:
            pd.update({'individual': predictions})
        else:
            pd.update({
                'average': averaged_predictions,
                'individual': predictions
            })
        return pd

    def _is_categorical(self, feature):
        return (self.categorical_names is not None) and (feature in self.categorical_names)

    def _is_numerical(self, feature):
        return (self.categorical_names is None) or (feature not in self.categorical_names)

    def _build_explanation(self, response_method, method, kind, feature_names, pds):
        deciles_values, feature_values = [], []
        pd_values = [] if kind in [Kind.AVERAGE, Kind.BOTH] else None
        ice_values = [] if kind in [Kind.INDIVIDUAL, Kind.BOTH] else None

        for pd in pds:
            feature_values.append(pd['values'])
            deciles_values.append(pd['deciles'])

            if Kind.AVERAGE in pd:
                pd_values.append(pd[Kind.AVERAGE])
            if Kind.INDIVIDUAL in pd:
                ice_values.append(pd[Kind.INDIVIDUAL])

        data = copy.deepcopy(DEFAULT_DATA_PD)
        data.update(
            all_feature_names=self.feature_names,
            all_categorical_names=self.categorical_names,
            all_target_names=self.target_names,
            feature_names=feature_names,
            feature_values=feature_values,
            ice_values=ice_values,
            pd_values=pd_values,
            deciles_values=deciles_values,
            response_method=response_method,
            method=method,
            kind=kind
        )
        return Explanation(meta=copy.deepcopy(self.meta), data=data)


    def reset_predictor(self, predictor: Any) -> None:
        pass



def plot_pd(exp: Explanation,
            features_list: Union[List[int], Literal['all']] = 'all',
            target_idx: Union[int, str] = 0,
            n_cols: int = 3,
            centered: bool = True,
            ax: Union['plt.Axes', np.ndarray, None] = None,
            sharey: str = 'all',
            pd_num_kw: Optional[dict] = None,
            ice_num_kw: Optional[dict] = None,
            pd_cat_kw: Optional[dict] = None,
            ice_cat_kw: Optional[dict] = None,
            pd_num_num_kw: Optional[dict] = None,
            pd_num_cat_kw: Optional[dict] = None,
            pd_cat_cat_kw: Optional[dict] = None,
            fig_kw: Optional[dict] = None) -> 'np.ndarray':
    """
    Plot Partial Dependence curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.
    features_list
        A list of features for which to plot the Partial Dependence curves or ``'all'`` for all features.
        Can be a integers denoting feature index denoting entries in `exp.feature_names`. Defaults to ``'all'``.
    target_idx
        Target index for which to plot the Partial Dependence curves. Can be a mix of integers denoting target
        index or strings denoting entries in `exp.target_names`.
    n_cols
        Number of columns to organize the resulting plot into.
    centered
        Boolean flag to center numerical Individual Conditional Expectation curves.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    sharey
        A parameter specifying whether the y-axis of the ALE curves should be on the same scale
        for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``.
    pd_num_kw
        Keyword arguments passed to the `plt.plot` function when plotting the Partial Dependenc.
    ice_num_kw
        Keyward arguments passed to the `plt.plot` function when plotting the Individual Conditional Expectation.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    An array of `matplotlib` axes with the resulting Partial Dependence plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}
    fig_kw = {**default_fig_kw, **fig_kw}

    if features_list == 'all':
        features_list = range(0, len(exp.feature_names))
    else:
        for features in features_list:
            if features > len(exp.feature_names):
                raise ValueError(f'The feature_list indices must be less than the '
                                 f'len(exp.feature_names) = {len(exp.feature_names)}. Received {features}.')

    # corresponds to the number of subplots
    n_features = len(features_list)

    # make axes
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_features != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_features)
        n_rows = math.ceil(n_features / n_cols)
        print(n_rows, n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)

        def _set_common_axes(start: int, stop: int, all_different: bool = False ):
            common_axes = None
            for i, spec in zip(range(start, stop), list(gs)[start:stop]):
                if not isinstance(exp.feature_names[i], Tuple) and (not all_different):
                    axes_ravel[i] = fig.add_subplot(spec, sharey=common_axes)
                    if common_axes is None:
                        common_axes = axes_ravel[i]
                else:
                    axes_ravel[i] = fig.add_subplot(spec)

        if sharey == 'all':
            _set_common_axes(0, n_features)
        elif sharey == 'row':
            for i in range(n_rows):
                _set_common_axes(i * n_cols, min((i + 1) * n_cols, n_features))
        else:
            _set_common_axes(0, n_features, all_different=True)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_features:
            raise ValueError(f"Expected ax to have {n_features} axes, got {ax.size}")
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    def _is_categorical(feature):
        feature_idx = np.where(exp.all_feature_names == feature)[0].item()
        return feature_idx in exp.all_categorical_names

    # make plots
    for ix, features, ax_ravel in zip(count(), features_list, axes_ravel):
        # extract the feature names
        feature_names = exp.feature_names[features]

        # if it is tuple, then we need a 2D plot and address 4 cases: (num, num), (num, cat), (cat, num), (cat, cat)
        if isinstance(feature_names, Tuple):
            f0, f1 = feature_names

            if (not _is_categorical(f0)) and (not _is_categorical(f1)):
                _ = _plot_two_pd_num_num(exp=exp,
                                         feature=features,
                                         target_idx=target_idx,
                                         ax=ax_ravel,
                                         pd_num_num_kw=pd_num_num_kw)

            elif _is_categorical(f0) and _is_categorical(f1):
                _ = _plot_two_pd_cat_cat(exp=exp,
                                         feature=features,
                                         target_idx=target_idx,
                                         ax=ax_ravel,
                                         pd_cat_cat_kw=pd_cat_cat_kw)

            else:
                _ = _plot_two_pd_num_cat(exp=exp,
                                         feature=features,
                                         target_idx=target_idx,
                                         ax=ax_ravel,
                                         pd_num_cat_kw=pd_num_cat_kw)

        else:
            if _is_categorical(feature_names):
                _ = _plot_one_pd_cat(exp=exp,
                                     feature=features,
                                     target_idx=target_idx,
                                     ax=ax_ravel,
                                     pd_cat_kw=pd_cat_kw,
                                     ice_cat_kw=ice_cat_kw)
            else:
                _ = _plot_one_pd_num(exp=exp,
                                     feature=features,
                                     target_idx=target_idx,
                                     centered=centered,
                                     ax=ax_ravel,
                                     pd_num_kw=pd_num_kw,
                                     ice_num_kw=ice_num_kw, )

    fig.set(**fig_kw)
    return axes


def _plot_one_pd_num(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     centered: bool = True,
                     ax: 'plt.Axes' = None,
                     pd_num_kw: dict = None,
                     ice_num_kw: dict = None) -> 'plt.Axes':
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    if exp.kind == Kind.AVERAGE:
        default_pd_num_kw = {'markersize': 2, 'marker': 'o', 'label': None}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}
        ax.plot(exp.feature_values[feature], exp.pd_values[feature][target_idx], **pd_num_kw)
        # shay = ax.get_shared_y_axes()
        # shay.remove(ax)

    elif exp.kind == Kind.INDIVIDUAL:
        default_ice_graph_kw = {'color': 'lightsteelblue', 'label': None}
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}

        # extract and center ice values if necessary
        ice_values = exp.ice_values[feature][target_idx].T
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(exp.feature_values[feature], ice_values, **ice_num_kw)
    else:
        default_pd_num_kw = {'linestyle': '--', 'linewidth': 2, 'color': 'tab:orange', 'label': 'average'}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}

        default_ice_graph_kw = {'alpha': 0.8, 'color': 'lightsteelblue', 'label': None}
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}

        # extract and center pd values if necessary
        pd_values = exp.pd_values[feature][target_idx]
        if centered:
            pd_values = pd_values - pd_values[0]

        # extract and center ice values if necessary
        ice_values = exp.ice_values[feature][target_idx].T
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(exp.feature_values[feature], ice_values, **ice_num_kw)
        ax.plot(exp.feature_values[feature], pd_values, **pd_num_kw)
        ax.legend()

    # add deciles markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.deciles_values[feature][1:], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.all_feature_names[feature])
    ax.set_ylabel(exp.all_target_names[target_idx])
    return ax


def _plot_one_pd_cat(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     ax: 'plt.Axes' = None,
                     pd_cat_kw: dict = None,
                     ice_cat_kw: dict = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        ax = plt.gca()

    feature_names = exp.feature_names[feature]
    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx] if (exp.pd_values is not None) else None
    ice_values = exp.ice_values[feature][target_idx] if (exp.ice_values is not None) else None

    def _get_feature_idx(feature):
        return np.where(exp.all_feature_names == feature)[0].item()

    # extract labels for the categorical feature
    feature_index = _get_feature_idx(feature_names)
    labels = [exp.all_categorical_names[feature_index][i] for i in feature_values.astype(np.int32)]

    if exp.kind in [Kind.INDIVIDUAL, Kind.BOTH]:
        x, y = [], []

        for i in range(len(feature_values)):
            x.append([feature_values[i]] * ice_values.shape[0])
            y.append(ice_values[:, i])

        x = np.concatenate(x, axis=0).astype(np.int32)
        y = np.concatenate(y, axis=0)

    if exp.kind == Kind.AVERAGE:
        default_pd_graph_kw = {'color': 'tab:blue'}
        pd_cat_kw = default_pd_graph_kw if pd_cat_kw is None else {**default_pd_graph_kw, **pd_cat_kw}
        sns.barplot(x=feature_values, y=pd_values, ax=ax, **pd_cat_kw)

    elif exp.kind == Kind.INDIVIDUAL:
        default_ice_cat_kw = {'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}
        sns.stripplot(x=x, y=y, ax=ax, **ice_cat_kw)

    else:
        default_pd_cat_kw = {'color': 'tab:orange', 'label': 'average'}
        pd_cat_kw = default_pd_cat_kw if pd_cat_kw is None else {**default_pd_cat_kw, **pd_cat_kw}

        default_ice_cat_kw = {'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}

        sns.barplot(x=feature_values, y=pd_values, ax=ax, **pd_cat_kw)
        sns.stripplot(x=x, y=y, ax=ax, **ice_cat_kw)
        ax.legend()

    # set xticks labels
    ax.set_xticklabels(labels)

    # set axis labels
    ax.set_xlabel(feature_names)
    ax.set_ylabel(exp.all_target_names[target_idx])
    return ax


def _plot_two_pd_num_num(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_num_num_kw: dict = None,):
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot Partial Dependence for kind='average'.")

    if ax is None:
        ax = plt.gca()

    # set contour plot default params
    default_pd_num_num_kw = {"alpha": 0.75}
    pd_num_num_kw = default_pd_num_num_kw if pd_num_num_kw is None else {**default_pd_num_num_kw, **pd_num_num_kw}

    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx]

    X, Y = np.meshgrid(feature_values[0], feature_values[1])
    Z = pd_values.T
    Z_level = np.linspace(Z.min(), Z.max(), 8)

    CS = ax.contour(X, Y, Z, levels=Z_level, linewidths=0.5, colors="k")
    ax.contourf(X, Y, Z, levels=Z_level, vmax=Z_level[-1], vmin=Z_level[0], **pd_num_num_kw)
    ax.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)

    # create the deciles line for the vertical & horizontal axis
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # the horizontal lines do not display (same for the sklearn)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.deciles_values[feature][0][1:], 0, 0.05, transform=trans)
    ax.hlines(exp.deciles_values[feature][1][1:], 0, 0.05, transform=trans)

    # reset xlim and ylim since they are overwritten by hlines and vlines
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # set x & y labels
    ax.set_xlabel(exp.feature_names[feature][0])
    ax.set_ylabel(exp.feature_names[feature][1])


def _plot_two_pd_num_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_num_cat_kw: dict = None):
    import matplotlib.pyplot as plt

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot Partial Dependece for kind='average'.")

    if ax is None:
        ax = plt.gca()

    def _get_feature_idx(feature):
        return np.where(exp.all_feature_names == feature)[0].item()

    def _is_categorical(feature):
        feature_idx = _get_feature_idx(feature)
        return feature_idx in exp.all_categorical_names

    # extract feature values and partial dependence values
    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx]

    # find which feature is categorical and which one is numerical
    feature_names = exp.feature_names[feature]
    if _is_categorical(feature_names[0]):
        feature_names = feature_names[::-1]
        feature_values = feature_values[::-1]
        pd_values = pd_values.T

    # define labels
    cat_feature_index = _get_feature_idx(feature_names[1])
    labels = [exp.all_categorical_names[cat_feature_index][i] for i in feature_values[1].astype(np.int32)]

    # plot lines
    default_pd_num_cat_kw = {'markersize': 2, 'marker': 'o'}
    pd_num_cat_kw = default_pd_num_cat_kw if pd_num_cat_kw is None else {**default_pd_num_cat_kw, **pd_num_cat_kw}
    ax.plot([], [], ' ', label=feature_names[1])

    for i in range(pd_values.shape[1]):
        x, y = feature_values[0], pd_values[:, i]
        pd_num_cat_kw.update({'label': labels[i]})
        ax.plot(x, y, **pd_num_cat_kw)

    ax.set_ylabel(exp.all_target_names[target_idx])
    ax.set_xlabel(feature_names[0])
    ax.legend()


def _plot_two_pd_cat_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_cat_cat_kw: dict = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        ax = plt.gca()

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot Partial Dependence for kind='average'.")

    def _get_feature_idx(feature):
        return np.where(exp.all_feature_names == feature)[0].item()

    feature_names = exp.feature_names[feature]
    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx]

    # extract labels for each categorical features
    feature0_index = _get_feature_idx(feature_names[0])
    feature1_index = _get_feature_idx(feature_names[1])
    labels0 = [exp.all_categorical_names[feature0_index][i] for i in feature_values[0].astype(np.int32)]
    labels1 = [exp.all_categorical_names[feature1_index][i] for i in feature_values[1].astype(np.int32)]

    # plot heatmap
    default_pd_cat_cat_kw = {
        'annot': True,
        'fmt': '.2f',
        'linewidths': .5,
        'yticklabels': labels0,
        'xticklabels': labels1
    }
    pd_cat_cat_kw = default_pd_cat_cat_kw if pd_cat_cat_kw is None else {**default_pd_cat_cat_kw, **pd_cat_cat_kw}
    sns.heatmap(pd_values, ax=ax, **pd_cat_cat_kw)

    # set ticks labels
    ax.set_xticklabels(labels1)
    ax.set_yticklabels(labels0)

    # set axis labels
    ax.set_xlabel(exp.feature_names[feature][1])
    ax.set_ylabel(exp.feature_names[feature][0])