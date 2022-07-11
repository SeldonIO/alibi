import copy
import numbers

import numpy as np
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal

from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier, is_regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import BaseHistGradientBoosting
from sklearn.inspection._partial_dependence import _grid_from_X
from sklearn.utils.extmath import cartesian
from sklearn.utils import _get_column_indices, _safe_indexing

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
            Note that for the faster `method='recursion'` option the only compatible paramter value is
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

        # # set the target_names when the user did not provide the target names
        # if self.target_names is None:
        #     pred = np.atleast_2d(self.predictor(X[0].reshape(1, -1)))
        #     n_targets = pred.shape[1]
        #     self.target_names = [f'c_{i}' for i in range(n_targets)]

        self.feature_names = np.array(self.feature_names)
        # self.target_names = np.array(self.target_names)
        self.categorical_names = {k: np.array(v) for k, v in self.categorical_names.items()}

        # parameters sanity checks
        self._params_sanity_checks(estimator=self.predictor, response_method=response_method, method=method, kind=kind)
        self._features_sanity_checks(features=features_list)

        explanation = []
        for features in features_list:
            pd = self._partial_dependence(estimator=self.predictor,
                                          X=X,
                                          features=features,
                                          response_method=response_method,
                                          percentiles=percentiles,
                                          grid_resolution=grid_resolution,
                                          method=method,
                                          kind=kind)
            explanation.append(pd)

        return self._build_explanation(explanation)

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

    def _partial_dependence(self,
                            estimator: Callable[[np.ndarray], np.ndarray],
                            X: np.ndarray,
                            features: Union[int, Tuple[int, int]],
                            response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                            percentiles: Tuple[float, float] = (0.05, 0.95),
                            grid_resolution: int = 100,
                            method: Literal['auto', 'recursion', 'brute'] = 'auto',
                            kind: Literal['average', 'individual', 'both'] = 'average'):

        if isinstance(features, Tuple):
            f0, f1 = features

            if self._is_numerical(f0) and self._is_numerical(f1):
                feature_indices = np.asarray(_get_column_indices(X, features), dtype=np.int32, order='C').ravel()
                grid, values = _grid_from_X(X=_safe_indexing(X, feature_indices, axis=1),
                                            percentiles=percentiles,
                                            grid_resolution=grid_resolution)

            elif self._is_numerical(f0) and self._is_categorical(f1):
                feature_indices0 = np.asarray(_get_column_indices(X, f0), dtype=np.int32, order='C').ravel()
                grid0, values0 = _grid_from_X(X=_safe_indexing(X, feature_indices0, axis=1),
                                              percentiles=percentiles,
                                              grid_resolution=grid_resolution)

                feature_indices1 = np.asarray(_get_column_indices(X, f1), dtype=np.int32, order='C').ravel()
                grid1, values1 = _grid_from_X(X=_safe_indexing(X, feature_indices1, axis=1),
                                              percentiles=percentiles,
                                              grid_resolution=np.inf)

                grid = cartesian(grid0.reshape(-1), grid1.reshape(-1))
                values = values0 + values1
            elif self._is_categorical(f0) and self._is_numerical(f1):
                feature_indices0 = np.asarray(_get_column_indices(X, f0), dtype=np.int32, order='C').ravel()
                grid0, values0 = _grid_from_X(X=_safe_indexing(X, feature_indices0, axis=1),
                                              percentiles=percentiles,
                                              grid_resolution=np.inf)

                feature_indices1 = np.asarray(_get_column_indices(X, f1), dtype=np.int32, order='C').ravel()
                grid1, values1 = _grid_from_X(X=_safe_indexing(X, feature_indices1, axis=1),
                                              percentiles=percentiles,
                                              grid_resolution=grid_resolution)

                grid = cartesian(grid0.reshape(-1), grid1.reshape(-1))
                values = values0 + values1
            else:

                grid, values = _grid_from_X(X=X[: [f0, f1]], percentiles=percentiles, grid_resolution=np.inf)
        else:
            feature_indices = np.asarray(_get_column_indices(X, features), dtype=np.int32, order='C').ravel()
            grid, values = _grid_from_X(X=_safe_indexing(X, feature_indices, axis=1),
                                        percentiles=percentiles,
                                        grid_resolution=grid_resolution if self._is_numerical(features) else np.inf)

        print("Grid:")
        print("=====")
        print(grid)

        print("Values:")
        print("=======")
        print(values)
        return None

    def _is_categorical(self, feature):
        return (self.categorical_names is not None) and (feature in self.categorical_names)

    def _is_numerical(self, feature):
        return (self.categorical_names is None) or (feature not in self.categorical_names)

    def _build_explanation(self, explanation):
        return explanation

    def reset_predictor(self, predictor: Any) -> None:
        pass