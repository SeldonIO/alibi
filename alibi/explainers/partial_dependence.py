import copy
import logging
import math
import numbers
from enum import Enum
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple, Type,
                    Union, no_type_check)

import matplotlib.pyplot as plt
import numpy as np
from alibi.api.defaults import DEFAULT_DATA_PD, DEFAULT_META_PD
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.ale import get_quantiles
from alibi.explainers.similarity.grad import get_options_string
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import \
    BaseHistGradientBoosting
from sklearn.inspection._partial_dependence import (
    _grid_from_X, _partial_dependence_brute, _partial_dependence_recursion)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.utils.extmath import cartesian
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class ResponseMethod(str, Enum):
    """ Enumeration of supported response_method. """
    AUTO = 'auto'
    PREDICT_PROBA = 'predict_proba'
    DECISION_FUNCTION = 'decision_function'


class Method(str, Enum):
    """ Enumeration of supported method. """
    AUTO = 'auto'
    RECURSION = 'recursion'
    BRUTE = 'brute'


class Kind(str, Enum):
    """ Enumeration of supported kind. """
    AVERAGE = 'average'
    INDIVIDUAL = 'individual'
    BOTH = 'both'


class PartialDependence(Explainer):
    def __init__(self,
                 predictor: Union[BaseEstimator, Callable[[np.ndarray], np.ndarray]],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 predictor_kw: Optional[Dict[str, Any]] = None):
        """
        Partial dependence for tabular datasets. Supports one feature or two feature interactions.

        Parameters
        ----------
        predictor
            A `sklearn` estimator or a prediction function which receives as input a `numpy` array and outputs a
            `numpy` array.
        feature_names
            A list of feature names used for displaying results.
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature.
        target_names
            A list of target/output names used for displaying results.
        predictor_kw
            Predictor identifier arguments when the predictor is a callable prediction function. The following
            arguments must be provided:

             - ``'predictor_type'`` : ``str`` - Type of the predictor. Available
             values: ``'regressor'`` | ``'classifier'``.

             - ``'prediction_fn'`` : ``str`` - Name of the prediction function. Available value for
             regression: ``'predict'``. Available values for
             classification: ``'predict_proba'`` | ``'decision_function'``. The choice should be considered
             in analogy with the `sklearn` estimators API and the `response_method` used in
             :py:meth:`alibi.explainers.partial_dependence.explain`.

            - ``'num_classes'`` : ``Optional[int]`` - Number of classes predicted by the `predictor` function.
            Considered only for ``prediction_type='classification'``.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PD))
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.target_names = target_names

        if isinstance(predictor, BaseEstimator):
            self.predictor = predictor
        elif callable(predictor):
            if predictor_kw is None:
                predictor_kw = {}
            self.predictor = PDEstimatorWrapper(predictor=predictor, **predictor_kw)
        else:
            raise ValueError(f'Unknown predictor type. Received {type(predictor)}. Supported type are: '
                             f'sklearn.base.BaseEstimator | Callable[[np.ndarray], np.ndarray].')

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                features_list: Optional[List[Union[int, Tuple[int, int]]]] = None,
                response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                method: Literal['auto', 'recursion', 'brute'] = 'auto',
                kind: Literal['average', 'individual', 'both'] = 'average',
                percentiles: Tuple[float, float] = (0.05, 0.95),
                grid_resolution: int = 100,
                grid_points: Optional[Dict[int, np.ndarray]] = None) -> Explanation:
        """
        Calculate the partial dependence for each feature with respect to the given target and the dataset `X`.

        Parameters
        ----------
        X
            An `N x F` tabular dataset used to calculate partial dependence curves. This is typically the training
            dataset or a representative sample.
        features_list
            An optional list of features or pairs of features for which to calculate the partial dependence for.
            If not provided, the partial dependence will be computed for all single features in the dataset.
        response_method
            Specifies the prediction function to be used. For classifier it specifies whether to use the
            `predict_proba` or the `decision_function`. For a regressor, the parameter is ignored. If set to `auto`,
            the `predict_proba` is tried first, and if not supported then it reverts to `decision_function`. Note
            that if `method='recursion'`, the that the prediction function always uses `decision_function`.
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
            If set to `'average'`, then only the partial dependence (PD) averaged across all samples from the dataset
            is returned. If set to `individual`, then only the Individual Conditional Expectation (ICE) is returned for
            each individual from the dataset. Otherwise, if set to `'both'`, then both the PD and the ICE are returned.
            Note that for the faster `method='recursion'` option the only compatible parameter value is
            `kind='average'`. To plot the ICE, consider using the more computation intensive `method='brute'`.
        percentiles
            Lower and upper percentiles used to create extreme values which can potential remove outliers in low
            density regions. The values must be in [0, 1].
        grid_resolution
            Number of equidistant points to split the range of each target feature. Only applies if the number of
            unique values of a target feature in the reference dataset `X` is less than the `grid_resolution` value.
        grid_points
            Custom grid points. Must be a `dict` where the keys are the target features indices and the values are
            monotonically increasing `numpy` arrays defining the grid points for numerical feature, and
            a subset of categorical feature values for a categorical feature. If the `grid_points` are not specified,
            then the grid will be constructed based on the unique target feature values available in the reference
            dataset `X`, or based on the `grid_resolution` and `percentiles` (check `grid_resolution` to see when
            it applies). For categorical features, the corresponding value in the `grid_points` can be
            specified either as `numpy` array of strings or `numpy` array of integers corresponding the label
            encodings. Note that the label encoding must match the ordering of the values provided in the
            `categorical_names`.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated partial Dependece
            curves. See usage at `Partial dependence examples`_ for details

            .. _Partial dependence examples:
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

        self.feature_names = np.array(self.feature_names)  # type: ignore
        self.categorical_names = {k: np.array(v) for k, v in self.categorical_names.items()}  # type: ignore

        # sanity checks
        self._grid_points_sanity_checks(grid_points=grid_points, n_features=n_features)
        self._features_sanity_checks(features=features_list)
        response_method, method, kind = self._params_sanity_checks(estimator=self.predictor,  # type: ignore
                                                                   response_method=response_method,
                                                                   method=method,
                                                                   kind=kind)

        # construct feature_names based on the feature_list. If feature_list is None, then initialize
        # feature_list with all single feature available in the dataset.
        if features_list:
            feature_names = np.array([tuple([self.feature_names[f] for f in features]) # type: ignore
                                      if isinstance(features, tuple) else self.feature_names[features]  # type: ignore
                                      for features in features_list], dtype=object)
        else:
            feature_names = self.feature_names  # type: ignore
            features_list = list(range(n_features))

        # buffer of all partial dependencies
        pds = []

        # compute partial dependencies for every features.
        # TODO: implement parallel version - future work as it can be done for ALE too
        for features in features_list:
            pds.append(
                self._partial_dependence(
                    estimator=self.predictor,
                    X=X,
                    features=features,
                    response_method=response_method,
                    method=method,
                    kind=kind,
                    percentiles=percentiles,
                    grid_resolution=grid_resolution,
                    grid_points=grid_points
                )
            )

        # set the target_names when the user did not provide the target names
        # we do it here to avoid checking model's type, prediction function etc
        if self.target_names is None:
            key = Kind.AVERAGE if kind in [Kind.AVERAGE, Kind.BOTH] else Kind.INDIVIDUAL
            n_targets = pds[0][key].shape[0]
            self.target_names = [f'c_{i}' for i in range(n_targets)]

        return self._build_explanation(response_method=response_method,
                                       method=method,
                                       kind=kind,
                                       feature_names=feature_names,  # type: ignore
                                       pds=pds)

    def _grid_points_sanity_checks(self, grid_points: Optional[Dict[int, np.ndarray]], n_features: int):
        """
        Grid points sanity checks.

        Parameters
        ----------
        grid_points
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain`.
        n_features
            Number of features in the dataset.
        """
        if grid_points is None:
            return

        if not np.all(np.isin(list(grid_points.keys()),  np.arange(n_features))):
            raise ValueError('The features provided in grid_points are not a subset of the dataset features.')

        for f in grid_points:
            if self._is_numerical(f):
                grid_points[f] = np.sort(grid_points[f])

            else:
                grid_points[f] = np.unique(grid_points[f])
                message = 'The grid points provided for the categorical feature {} are invalid. '\
                          'For categorical features, the grid points must be a subset of the features '\
                          'values defined in categorical_names. Received an unknown value of {}'

                # convert to label encoding if the grid is provided as strings
                if grid_points[f].dtype.type is np.str_:
                    int_values = []

                    for str_vals in grid_points[f]:
                        # self.categorical_names cannot be None because of the check in self._is_numerical
                        index = np.where(str_vals == self.categorical_names[f])[0]  # type: ignore[index]
                        if len(index) > 0:
                            int_values.append(index[0])
                        else:
                            raise ValueError(message.format(f, str_vals))

                    grid_points[f] = np.array(int_values)

                # self.categorical_names cannot be None because of the check in self._is numerical
                mask = np.isin(grid_points[f], np.arange(len(self.categorical_names[f])))  # type: ignore[index]
                if not np.all(mask):
                    index = np.where(not mask)[0][0]
                    raise ValueError(message.format(f, grid_points[f][index]))

    def _features_sanity_checks(self, features: Optional[List[Union[int, Tuple[int, int]]]]) -> None:
        """
        Features sanity checks.

        Parameters
        ----------
        features
            List of feature indices or pairs of feature indices to compute the partial dependence for.
        """
        if features is None:
            return

        def check_feature(f):
            if not isinstance(f, numbers.Integral):
                raise ValueError(f'All feature entries must be integers. Got a feature value of {type(f)} type.')
            if f >= len(self.feature_names):
                raise ValueError(f'All feature entries must be less than len(feature_names)={len(self.feature_names)}. '
                                 f'Got a feature value of {f}.')
            if f < 0:
                raise ValueError(f'All features entries must be greater or equal to 0. Got a feature value of {f}.')

        for f in features:
            if isinstance(f, tuple):
                if len(f) != 2:
                    raise ValueError(f'Current implementation of the Partial dependence supports only two features '
                                     f'at a time when a tuple is passed. Received {len(f)} features with the '
                                     f'values {f}.')

                check_feature(f[0])
                check_feature(f[1])
            else:
                check_feature(f)

    def _params_sanity_checks(self,
                              estimator: BaseEstimator,
                              response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                              method: Literal['auto', 'recursion', 'brute'] = 'auto',
                              kind: Literal['average', 'individual', 'both'] = 'average'
                              ) -> Tuple[str, str, str]:
        """
        Parameters sanity checks. Most of the code is borrowed from:
        https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/inspection/_partial_dependence.py

        estimator
            A `sklearn` estimator.
        response_method, method, kind
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.

        Returns
        -------
        Update parameters `(response_method, method, kind)`.
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
            method = Method.BRUTE  # type: ignore

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
                            estimator: BaseEstimator,
                            X: np.ndarray,
                            features: Union[int, Tuple[int, int]],
                            response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                            method: Literal['auto', 'recursion', 'brute'] = 'auto',
                            kind: Literal['average', 'individual', 'both'] = 'average',
                            percentiles: Tuple[float, float] = (0., 1.),
                            grid_resolution: int = 100,
                            grid_points: Optional[Dict[int, np.ndarray]] = None
                            ) -> Dict[str, np.ndarray]:
        """
        Computes partial dependence for a feature or a pair of features.

        Parameters
        ----------
        estimator, X, features, response_method, method, kind, percentiles, grid_resolution, grid_points
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.

        Returns
        -------
        A dictionary containing the feature(s) values, feature(s) deciles, average and/or individual values
        (i.e. partial dependence or individual conditional expectation) of the give (pair) of feature(s))
        """
        if isinstance(features, numbers.Integral):
            features = (features, )

        if grid_points is None:
            grid_points = {}

        deciles, values, features_indices = [], [], [],
        for f in features:  # type: ignore
            # check against the category map and raise a warning. Need to choose one or the other.
            # Note that we are using the _safe_indexing procedure implement in sklearn to retrieve the column.
            # This is because the _safe_indexing supports many data types such as sparse matrix representation,
            # pandas dataframes etc. In case we would like to enhance alibi in the future to support other
            # data types than np.ndarrays, this can be useful.
            f_indices = np.asarray(_get_column_indices(X, f), dtype=np.int32, order='C').ravel()
            X_f = _safe_indexing(X, f_indices, axis=1)

            # get deciles for the current feature
            deciles_f = get_quantiles(X_f, num_quantiles=11) if self._is_numerical(f) else None

            if f not in grid_points:
                # construct grid for feature f. Note that for categorical features we pass the
                # grid resolution to be infinity because otherwise we risk to apply `linspace` to
                # categorical values, which does not make sense.
                _, values_f = _grid_from_X(X_f,
                                           percentiles=percentiles,
                                           grid_resolution=grid_resolution if self._is_numerical(f) else np.inf)
            else:
                values_f = [grid_points[f_indices[0]]]

            features_indices.append(f_indices)
            deciles.append(deciles_f)
            values += values_f

        # covers also the case of a single feature, just to ensure it has the right shape
        features_indices = np.concatenate(features_indices, axis=0)
        grid = cartesian(tuple([v.reshape(-1) for v in values]))

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

        # special case when dealing with a binary classifier that uses predict_proba - sklearn remove
        # the computation for the label 0 and only returns the predictions for label 1. To make it consistent with
        # the ALE, we re-add the predictions for label 0.
        n_outputs = averaged_predictions.shape[0]
        conditions = [
            is_classifier(estimator),
            response_method in [ResponseMethod.PREDICT_PROBA, ResponseMethod.AUTO],
            hasattr(estimator, 'predict_proba'),  # works because predict_proba has priority over decision_function
            n_outputs == 1
        ]

        if all(conditions):
            averaged_predictions = np.concatenate([1 - averaged_predictions, averaged_predictions], axis=0)
            if method == 'brute':
                predictions = np.concatenate([1 - predictions, predictions], axis=0)

        # define feature values (i.e. grid values) and the corresponding deciles. Note that the deciles
        # were computed on the raw (i.e. unprocessed) feature value as provided in the reference dataset X
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

    def _is_categorical(self, feature) -> bool:
        """
        Checks if the given feature is categorical.

        Parameters
        ----------
        feature
            Feature to be checked.

        Returns
        -------
        ``True`` if the feature is categorical. ``False`` otherwise.
        """
        return (self.categorical_names is not None) and (feature in self.categorical_names)

    def _is_numerical(self, feature):
        """
        Checks if the given feature is numerical.

        Parameters
        ----------
        feature
            Feature to be checked.

        Returns
        -------
        ``True`` if the feature is numerical. ``False`` otherwise.
        """
        return (self.categorical_names is None) or (feature not in self.categorical_names)

    def _build_explanation(self,
                           response_method: str,
                           method: str,
                           kind: str,
                           feature_names: List[Union[int, Tuple[int, int]]],
                           pds: List[Dict[str, np.ndarray]]) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        response_method, method, kind
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.
        feature_names
            List of feature of pairs of features for which the partial dependencies/individual conditional expectation
            were computed.
        pds
            List of dictionary containing the partial dependencies/individual conditional expectation. For
            more details see :py:meth:`alibi.explainers.partial_dependence.PartialDependence._partial_dependence`.

        Returns
        -------
        `Explanation` object.
        """
        deciles_values, feature_values = [], []
        pd_values = [] if kind in [Kind.AVERAGE, Kind.BOTH] else None  # type: Optional[List[np.ndarray]]
        ice_values = [] if kind in [Kind.INDIVIDUAL, Kind.BOTH] else None  # type: Optional[List[np.ndarray]]

        for pd in pds:
            feature_values.append(pd['values'])
            deciles_values.append(pd['deciles'])

            if (pd_values is not None) and (Kind.AVERAGE in pd):
                pd_values.append(pd[Kind.AVERAGE])
            if (ice_values is not None) and Kind.INDIVIDUAL in pd:
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
        """
        Resets the predictor function

        Parameters
        ----------
        predictor
            New `sklearn` estimator.
        """
        self.predictor = predictor


# No type check due to the generic explanation object
@no_type_check
def plot_pd(exp: Explanation,
            features_list: Union[List[int], Literal['all']] = 'all',
            target_idx: int = 0,
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
    Plot Partial dependence curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.
    features_list
        A list of features for which to plot the Partial dependence curves or ``'all'`` for all features.
        Can be a integers denoting feature index denoting entries in `exp.feature_names`. Defaults to ``'all'``.
    target_idx
        Target index for which to plot the Partial dependence curves. Can be a mix of integers denoting target
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
        Keyword arguments passed to the `matplotlib plot` function when plotting the partial dependence
        for a numerical feature.
    ice_num_kw
        Keyword arguments passed to the `matplotlib plot` function when plotting the individual conditional
        expectation for a numerical feature.
    pd_cat_kw
        Keyword arguments passed to the `seaborn bar` function when plotting the partial dependence for a
        categorical feature.
    ice_cat_kw
        Keyword arguments passed to the `seaborn stripplot` function when plotting the individual conditional
        expectation for a categorical feature.
    pd_num_num_kw
        Keyword arguments passed to the `matplotlib contourf` function when plotting the partial dependence
        for two numerical features.
    pd_num_cat_kw
        Keyword arguments passed to the `matplotlib plot` function when plotting the partial dependence for
        a numerical and a categorical feature.
    pd_cat_cat_kw
        Keyword arguments passed to the `seaborn heatmap` functon when plotting the partial dependence for
        two categorical features.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    An array of `matplotlib` axes with the resulting Partial dependence plots.
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

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)

        def _set_common_axes(start: int, stop: int, all_different: bool = False):
            """ Helper function to add subplots and share common y axes. """
            common_axes = None
            for i, spec in zip(range(start, stop), list(gs)[start:stop]):
                if not isinstance(exp.feature_names[i], tuple) and (not all_different):
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
    for features, ax_ravel in zip(features_list, axes_ravel):  # type: ignore
        # extract the feature names
        feature_names = exp.feature_names[features]

        # if it is tuple, then we need a 2D plot and address 4 cases: (num, num), (num, cat), (cat, num), (cat, cat)
        if isinstance(feature_names, tuple):
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
                                     centered=centered,
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


# No type check due to the generic explanation object
@no_type_check
def _plot_one_pd_num(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     centered: bool = True,
                     ax: 'plt.Axes' = None,
                     pd_num_kw: Optional[dict] = None,
                     ice_num_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plots one way partial dependence curve for a single numerical feature.

    Parameters
    ----------
    exp, feature, target_idx, centered, pd_num_kw, ice_num_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.
    pd_num_kw
        Keyword arguments passed to the `matplotlib plot` function when plotting the partial dependence
        for a numerical feature.
    ice_num_kw
        Keyword arguments passed to the `matplotlib plot` function when plotting the individual conditional
        expectation for a numerical feature.

    Returns
    -------
    `matplotlib` axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx] if (exp.pd_values is not None) else None
    ice_values = exp.ice_values[feature][target_idx].T if (exp.ice_values is not None) else None

    if exp.kind == Kind.AVERAGE:
        default_pd_num_kw = {'markersize': 2, 'marker': 'o', 'label': None}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}
        ax.plot(feature_values, pd_values, **pd_num_kw)

    elif exp.kind == Kind.INDIVIDUAL:
        default_ice_graph_kw = {'color': 'lightsteelblue', 'label': None}
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}

        # extract and center ice values if necessary
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(feature_values, ice_values, **ice_num_kw)
    else:
        default_pd_num_kw = {'linestyle': '--', 'linewidth': 2, 'color': 'tab:orange', 'label': 'average'}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}

        default_ice_graph_kw = {'alpha': 0.6, 'color': 'lightsteelblue', 'label': None}  # type: ignore
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}

        # center pd values if necessary
        if centered:
            pd_values = pd_values - pd_values[0]

        # center ice values if necessary
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(feature_values, ice_values, **ice_num_kw)
        ax.plot(feature_values, pd_values, **pd_num_kw)
        ax.legend()

    # add deciles markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.deciles_values[feature][1:], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.feature_names[feature])
    ax.set_ylabel(exp.all_target_names[target_idx])
    return ax


# No type check due to the generic explanation object
@no_type_check
def _plot_one_pd_cat(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     centered: bool = True,
                     ax: 'plt.Axes' = None,
                     pd_cat_kw: Optional[dict] = None,
                     ice_cat_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plots one way partial dependence curve for a single categorical feature.

    Parameters
    ----------
    exp, feature, target_idx, pd_cat_kw, ice_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.
    pd_cat_kw
        Keyword arguments passed to the `seaborn bar` function when plotting the partial dependence for a
        categorical feature.
    ice_cat_kw
        Keyword arguments passed to the `seaborn stripplot` function when plotting the individual conditional
        expectation for a categorical feature.

    Returns
    -------
    `matplotlib` axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    feature_names = exp.feature_names[feature]
    feature_values = exp.feature_values[feature]
    pd_values = exp.pd_values[feature][target_idx] if (exp.pd_values is not None) else None
    ice_values = exp.ice_values[feature][target_idx].T if (exp.ice_values is not None) else None

    def _get_feature_idx(feature):
        return np.where(exp.all_feature_names == feature)[0].item()

    feature_index = _get_feature_idx(feature_names)
    labels = [exp.all_categorical_names[feature_index][i] for i in feature_values.astype(np.int32)]

    if exp.kind == Kind.AVERAGE:
        default_pd_graph_kw = {'markersize': 8, 'marker': 's', 'color': 'tab:blue'}
        pd_cat_kw = default_pd_graph_kw if pd_cat_kw is None else {**default_pd_graph_kw, **pd_cat_kw}
        ax.plot(labels, pd_values, **pd_cat_kw)

    elif exp.kind == Kind.INDIVIDUAL:
        default_ice_cat_kw = {'markersize': 4, 'marker': 's', 'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}

        # extract and center ice values if necessary
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(labels, ice_values, **ice_cat_kw)

    else:
        default_pd_cat_kw = {'markersize': 8, 'marker': 's', 'color': 'tab:orange', 'label': 'average'}
        pd_cat_kw = default_pd_cat_kw if pd_cat_kw is None else {**default_pd_cat_kw, **pd_cat_kw}

        default_ice_cat_kw = {'alpha': 0.6, 'markersize': 4, 'marker': 's', 'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}

        # center pd values if necessary
        if centered:
            pd_values = pd_values - pd_values[0]

        # center ice values if necessary
        if centered:
            ice_values = ice_values - ice_values[0:1]

        ax.plot(labels, ice_values, **ice_cat_kw)
        ax.plot(labels, pd_values, **pd_cat_kw)
        ax.legend()

    # rotate xticks labels
    ax.tick_params(axis='x', rotation=90)

    # set axis labels
    ax.set_xlabel(feature_names)
    ax.set_ylabel(exp.all_target_names[target_idx])
    return ax


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_num_num(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_num_num_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plots two ways partial dependence curve for two numerical features.

    Parameters
    ----------
    exp, feature, target_idx, pd_num_num_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.
    pd_num_num_kw
        Keyword arguments passed to the `matplotlib contourf` function when plotting the partial dependence
        for two numerical features.

    Returns
    -------
    `matplotlib` axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot partial dependence for kind='average'.")

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
    return ax


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_num_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_num_cat_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plots two ways partial dependence curve for a numerical feature and a categorical feature.

    Parameters
    ----------
    exp, feature, target_idx, pd_num_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.
    pd_num_cat_kw
        Keyword arguments passed to the `matplotlib plot` function when plotting the partial dependence for
        a numerical and a categorical feature.

    Returns
    -------
    `matplotlib` axes.
    """
    import matplotlib.pyplot as plt

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot partial dependece for kind='average'.")

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


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_cat_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: 'plt.Axes' = None,
                         pd_cat_cat_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plots two ways partial dependence curve for a numerical feature and a categorical feature.

    Parameters
    ----------
    exp, feature, target_idx, pd_cat_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.
    pd_cat_cat_kw
        Keyword arguments passed to the `seaborn heatmap` functon when plotting the partial dependence for
        two categorical features.

    Return
    ------
    `matplotlib` axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        ax = plt.gca()

    if exp.kind != Kind.AVERAGE:
        raise ValueError("Can only plot partial dependence for kind='average'.")

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
    return ax


class PredictorType(str, Enum):
    """ Enumeration of supported predictor types. """
    REGRESSION = 'regressor'
    CLASSIFIER = 'classifier'


class PredictionFunction(str, Enum):
    """ Enumeration of supported prediction function. """
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'
    DECISION_FUNCTION = 'decision_function'


class PDEstimatorWrapper:
    """ Estimator wrapper class for a black-box predictor to be compatible to the
    :py:class:`alibi.explainers.partial_dependence.PartialDependence`."""

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 predictor_type: Literal['regression', 'classifier'],
                 prediction_fn: Literal['predict', 'predict_proba', 'decision_function'],
                 num_classes: Optional[int] = None):
        """
        Constructor.

        Parameters
        ----------
        predictor
            Prediction function to be wrapped.
        predictor_type
            Type of the predictor. Available values: ``'regressor'`` | ``'classifier'``.
        prediction_fn
            Name of the prediction function. Available value for regression: ``'predict'``. Available values
            for classification: ``'predict_proba'`` | ``'decision_function'``. The choice should be considered
            in analogy with the `sklearn` estimators API and the `response_method` used in
            :py:meth:`alibi.explainers.partial_dependence.explain`.
        num_classes
            Number of classes predicted by the `predictor` function. Considered only for
            ``prediction_type='classification'``.
        """
        self.predictor = predictor
        self.is_fitted_ = True  # to check if fitted, sklearn also looks at vars that end in _ and do not start with __

        if predictor_type in PredictorType.__members__.values():
            self._estimator_type = predictor_type
        else:
            raise ValueError(f"predictor_type='{predictor_type}' is invalid. Accepted predictor_type names "
                             f"are {get_options_string(ResponseMethod)}.")

        if predictor_type == PredictorType.CLASSIFIER:
            if isinstance(num_classes, numbers.Integral):
                self.classes_ = np.arange(num_classes)
            else:
                raise ValueError(f"num_classes must be an integer when "
                                 f"predictor_type='{PredictorType.CLASSIFIER.value}'.")

            if prediction_fn == PredictionFunction.PREDICT_PROBA:
                self.predict_proba = predictor
            elif prediction_fn == PredictionFunction.DECISION_FUNCTION:
                self.decision_function = predictor
            else:
                raise ValueError(f"prediction_fn='{prediction_fn}' is invalid when "
                                 f"predictor_type='{PredictorType.CLASSIFIER.value}'. "
                                 f"Accepted prediction_fn names are {get_options_string(PredictionFunction)}.")

        else:
            if prediction_fn == PredictionFunction.PREDICT:
                self.predict = predictor
            else:
                raise ValueError(f"prediction_fn={prediction_fn} is invalid when "
                                 f"predictor_type='{PredictorType.REGRESSION}'. "
                                 f"Accepted predictor_type names are {get_options_string(PredictorType)}.")

    def __sklearn_is_fitted__(self):
        return self.is_fitted_

    def fit(self, *args, **kwargs):
        pass


# 1. TODO: consider a better `explain` API for PD and ALE -- maybe in the future ...
# 2. SOLVED: consider all one way as default for `explain -> features_list`. Check ALE -> solved
# 3. SOLVED: add custom grid_points
# 4. SOLVED : consider all values of a categorical features instead of using the unique values in the data?
# This was solved by custom grid_points
# 5. SOLVED: consider wrapping the black-box predictor inside init
# 6. SOLVED: decide whether the ICE for categorical are useful? What stories does it tell. Don't show how the output
# evolves for an individual. Line plot with markers might be a better option
# 7. SOLVED: display for both targets in binary classification?
# 8. TODO: OHE not supported for now, but do include decorator workaround in the example
