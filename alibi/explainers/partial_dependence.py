import copy
import logging
import math
import numbers
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import (Callable, Dict, Iterable, List, Optional, Tuple, Union,
                    no_type_check)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mquantiles
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import \
    BaseHistGradientBoosting
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.extmath import cartesian
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from alibi.api.defaults import DEFAULT_DATA_PD, DEFAULT_META_PD
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.ale import get_quantiles
from alibi.utils import _get_options_string

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


logger = logging.getLogger(__name__)


class Kind(str, Enum):
    """ Enumeration of supported kind. """
    AVERAGE = 'average'
    INDIVIDUAL = 'individual'
    BOTH = 'both'


class PartialDependenceBase(Explainer, ABC):
    def __init__(self,
                 predictor: Union[BaseEstimator, Callable[[np.ndarray], np.ndarray]],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Base class of the partial dependence for tabular datasets. Supports multiple feature interactions.

        Parameters
        ----------
        predictor
            A `sklearn` estimator or a prediction function which receives as input a `numpy` array of size `N x F`
            and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input
            instances, `F` is the number of features and `T` is the number of targets.
        feature_names
            A list of feature names used for displaying results.
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature. Necessary to
            identify the categorical features in the dataset. An example for `categorical_names` would be::

                category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]}

        target_names
            A list of target/output names used for displaying results.
        verbose
            Whether to print the progress of the explainer.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PD))
        self.predictor = predictor
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.target_names = target_names
        self.verbose = verbose

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                features: Optional[List[Union[int, Tuple[int, int]]]] = None,
                kind: Literal['average', 'individual', 'both'] = 'average',
                percentiles: Tuple[float, float] = (0., 1.),
                grid_resolution: int = 100,
                grid_points: Optional[Dict[int, Union[List, np.ndarray]]] = None) -> Explanation:
        """
        Calculates the partial dependence for each feature and/or tuples of features with respect to the all targets
        and the reference dataset `X`.

        Parameters
        ----------
        X
            A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the
            training dataset or a representative sample.
        features
            An optional list of features or tuples of features for which to calculate the partial dependence.
            If not provided, the partial dependence will be computed for every single features in the dataset.
            Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where
            ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
        kind
            If set to ``'average'``, then only the partial dependence (PD) averaged across all samples from the dataset
            is returned. If set to ``'individual'``, then only the individual conditional expectation (ICE) is
            returned for each data point from the dataset. Otherwise, if set to ``'both'``, then both the PD and
            the ICE are returned.
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
            curves. See usage at `Partial dependence examples`_ for details

            .. _Partial dependence examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html
        """
        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional.')

        # extract number of features
        n_features = X.shape[1]

        # set the `features_names` when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # set `categorical_names` when the user did not provide the category mapping
        if self.categorical_names is None:
            self.categorical_names = {}

        # sanity checks
        self._grid_points_sanity_checks(grid_points=grid_points, n_features=n_features)
        self._features_sanity_checks(features=features)

        # construct `feature_names` based on the `features`. If `features` is ``None``, then initialize
        # `features` with all single feature available in the dataset.
        if features:
            feature_names = [tuple([self.feature_names[f] for f in features])
                             if isinstance(features, tuple) else self.feature_names[features]
                             for features in features]
        else:
            feature_names = self.feature_names  # type: ignore[assignment]
            features = list(range(n_features))

        # compute partial dependencies for every features.
        # TODO: implement parallel version - future work as it can be done for ALE too
        pds = []

        for ifeatures in tqdm(features, disable=not self.verbose):
            pds.append(
                self._partial_dependence(
                    X=X,
                    features=ifeatures,
                    kind=kind,
                    percentiles=percentiles,
                    grid_resolution=grid_resolution,
                    grid_points=grid_points
                )
            )

        # extract the number of targets that the PD/ICE was computed for
        key = Kind.AVERAGE if kind in [Kind.AVERAGE, Kind.BOTH] else Kind.INDIVIDUAL
        n_targets = pds[0][key].shape[0]

        if self.target_names is None:
            # set the `target_names` when the user did not provide the target names
            # we do it here to avoid checking model's type, prediction function etc.
            self.target_names = [f'c_{i}' for i in range(n_targets)]

        elif len(self.target_names) != n_targets:
            logger.warning('The length of `target_names` does not match the number of predicted outputs. '
                           'Ensure that the lengths match, otherwise a call to the `plot_pd` method might '
                           'raise an error or produce undesired labeling.')

        # update `meta['params']` here because until this point we don't have the `target_names`
        self.meta['params'].update(kind=kind,
                                   percentiles=percentiles,
                                   grid_resolution=grid_resolution,
                                   feature_names=self.feature_names,
                                   categorical_names=self.categorical_names,
                                   target_names=self.target_names)

        return self._build_explanation(kind=kind,
                                       feature_names=feature_names,  # type: ignore[arg-type]
                                       pds=pds)

    def _grid_points_sanity_checks(self, grid_points: Optional[Dict[int, Union[List, np.ndarray]]], n_features: int):
        """
        Grid points sanity checks.

        Parameters
        ----------
        grid_points
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependenceBase.explain`.
        n_features
            Number of features in the dataset.
        """
        if grid_points is None:
            return

        if not np.all(np.isin(list(grid_points.keys()),  np.arange(n_features))):
            raise ValueError('The features provided in `grid_points` are not a subset of the dataset features.')

        for f in grid_points:
            if self._is_numerical(f):
                grid_points[f] = np.sort(grid_points[f])  # from this point onward, `grid_points[f]` is `np.ndarray`

            else:
                grid_points[f] = np.unique(grid_points[f])  # from this point onward, `grid_points[f]` is `np.ndarray`
                message = "The grid points provided for the categorical feature {} are invalid. "\
                          "For categorical features, the grid points must be a subset of the features "\
                          "values defined in `categorical_names`. Received an unknown value of '{}'."

                # convert to label encoding if the grid is provided as strings
                if grid_points[f].dtype.type is np.str_:  # type: ignore[union-attr]
                    int_values = []

                    for str_val in grid_points[f]:
                        try:
                            # `self.categorical_names` cannot be empty because of the check in `self._is_numerical`
                            index = self.categorical_names[f].index(str_val)  # type: ignore[index]
                        except ValueError:
                            raise ValueError(message.format(f, str_val))
                        int_values.append(index)
                    grid_points[f] = np.array(int_values)

                # `self.categorical_names` cannot be empty because of the check in `self._is numerical`
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
            List of feature indices or tuples of feature indices to compute the partial dependence for.
        """
        if features is None:
            return

        def check_feature(f):
            if not isinstance(f, numbers.Integral):
                raise ValueError(f'All feature entries must be integers. Got a feature value of {type(f)} type.')
            if f >= len(self.feature_names):
                raise ValueError(f'All feature entries must be less than '
                                 f'``len(feature_names)={len(self.feature_names)}``. Got a feature value of {f}.')
            if f < 0:
                raise ValueError(f'All feature entries must be greater or equal to 0. Got a feature value of {f}.')

        for feats in features:
            if not isinstance(feats, tuple):
                feats = (feats, )  # type: ignore[assignment]

            for f in feats:  # type: ignore[union-attr]
                check_feature(f)

    def _partial_dependence(self,
                            X: np.ndarray,
                            features: Union[int, Tuple[int, int]],
                            kind: Literal['average', 'individual', 'both'] = 'average',
                            percentiles: Tuple[float, float] = (0.05, 0.95),
                            grid_resolution: int = 100,
                            grid_points: Optional[Dict[int, Union[List, np.ndarray]]] = None
                            ) -> Dict[str, np.ndarray]:
        """
        Computes partial dependence for a feature or a tuple of features.

        Parameters
        ----------
        X, method, kind, percentiles, grid_resolution, grid_points
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependenceBase.explain` method.
        features
            A feature or tuples of features for which to calculate the partial dependence.

        Returns
        -------
        A dictionary containing the feature(s) values, feature(s) deciles, average and/or individual values
        (i.e. partial dependence or individual conditional expectation) for the given (tuple of) feature(s))
        """
        if isinstance(features, numbers.Integral):
            features = (features, )

        if grid_points is None:
            grid_points = {}

        deciles, values, features_indices = [], [], [],
        for f in features:  # type: ignore[union-attr]
            # extract column. TODO _safe_indexing in the future to support more input types.
            X_f = X[:, f]

            # get deciles for the current feature if the feature is numerical
            deciles_f = get_quantiles(X_f, num_quantiles=11) if self._is_numerical(f) else None

            if f not in grid_points:
                # construct grid for feature `f`. Note that for categorical features we pass the
                # grid resolution to be infinity because otherwise we risk to apply `linspace` to
                # categorical values, which does not make sense.
                values_f = self._grid_from_X(
                    X=X_f.reshape(-1, 1),
                    percentiles=percentiles,
                    grid_resolution=grid_resolution if self._is_numerical(f) else np.inf  # type: ignore[arg-type]
                )
            else:
                values_f = [grid_points[f]]

            features_indices.append(f)
            deciles.append(deciles_f)
            values += values_f

        # perform cartesian product between feature values. Covers also the case of a single feature.
        features_indices = np.array(features_indices, dtype=np.int32)  # type: ignore[assignment]
        grid = cartesian([v.reshape(-1) for v in values])

        # compute the PD and ICE - separate implementation for `PartialDependence` and `TreePartialDependence`
        averaged_predictions, predictions = self._compute_pd(grid=grid,
                                                             features=features_indices,  # type: ignore[arg-type]
                                                             X=X)

        # reshape `averaged_predictions` to (n_outputs, n_values_feature_0, n_values_feature_1, ...)
        averaged_predictions = averaged_predictions.reshape(-1, *[val.shape[0] for val in values])

        if predictions is not None:
            # reshape `predictions` to (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
            predictions = predictions.reshape(-1, X.shape[0], *[val.shape[0] for val in values])

        # define feature values (i.e. grid values) and the corresponding deciles. Note that the deciles
        # were computed on the raw (i.e. unprocessed) feature value as provided in the reference dataset `X`
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

    @abstractmethod
    def _compute_pd(self,
                    grid: np.ndarray,
                    features: np.ndarray,
                    X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes the PD and ICE.

        Parameters
        ----------
        grid
            Cartesian product between feature values. Covers also the case of a single feature.
        features
            Feature column indices.
        X
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependenceBase.explain`.

        Returns
        -------
        Tuple consisting of the PD and optionally the ICE.
        """
        raise NotImplementedError()

    def _grid_from_X(self, X: np.ndarray, percentiles: Tuple[float, float], grid_resolution: int):
        """
        Generate a grid of points based on the percentiles of `X`. If `grid_resolution` is bigger than the number
        of unique values in the jth column of `X`, then those unique values will be used instead.
        Code borrowed from:
        https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/inspection/_partial_dependence.py

        Parameters
        ----------
        X
            Array to generate the grid for.
        percentiles
            The percentiles which are used to construct the extreme values of the grid. Must be in [0, 1].
        grid_resolution
            The number of equally spaced points to be placed on the grid for each feature.

        Returns
        -------
        The values with which the grid has been created. The size of each array `values[j]` is either
        `grid_resolution`, or the number of unique values in `X[:, j]`, whichever is smaller.
        """
        if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
            raise ValueError("`percentiles` must be a sequence of 2 elements.")

        if not all(0 <= x <= 1 for x in percentiles):  # type: ignore[attr-defined]
            raise ValueError("`percentiles` values must be in [0, 1].")

        if percentiles[0] >= percentiles[1]:  # type: ignore[index]
            raise ValueError("`percentiles[0]` must be strictly less than `percentiles[1]`.")

        if grid_resolution <= 1:
            raise ValueError("`grid_resolution` must be strictly greater than 1.")

        values = []
        for feature in range(X.shape[1]):
            uniques = np.unique(X[:, feature])

            if uniques.shape[0] < grid_resolution:
                # feature has low resolution use unique vals
                axis = uniques
            else:
                # create axis based on percentiles and grid resolution
                emp_percentiles = mquantiles(X[:, feature], prob=percentiles, axis=0).data

                if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                    raise ValueError("`percentiles` are too close to each other, unable to build the grid. "
                                     "Please choose percentiles that are further apart.")

                # construct equidistant grid points
                axis = np.linspace(emp_percentiles[0], emp_percentiles[1], num=grid_resolution, endpoint=True)

            values.append(axis)
        return values

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
        return feature not in self.categorical_names

    def _build_explanation(self,
                           kind: str,
                           feature_names: List[Union[int, Tuple[int, int]]],
                           pds: List[Dict[str, np.ndarray]]) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        kind
            See :py:meth:`alibi.explainers.partial_dependence.PartialDependenceBase.explain` method.
        feature_names
            List of feature or tuples of features for which the partial dependencies/individual conditional
            expectation were computed.
        pds
            List of dictionary containing the partial dependencies/individual conditional expectation.

        Returns
        -------
        `Explanation` object.
        """
        feature_deciles, feature_values = [], []
        pd_values = [] if kind in [Kind.AVERAGE, Kind.BOTH] else None  # type: Optional[List[np.ndarray]]
        ice_values = [] if kind in [Kind.INDIVIDUAL, Kind.BOTH] else None  # type: Optional[List[np.ndarray]]

        for pd in pds:
            feature_values.append(pd['values'])
            feature_deciles.append(pd['deciles'])

            if (pd_values is not None) and (Kind.AVERAGE in pd):
                pd_values.append(pd[Kind.AVERAGE])
            if (ice_values is not None) and Kind.INDIVIDUAL in pd:
                ice_values.append(pd[Kind.INDIVIDUAL])

        data = copy.deepcopy(DEFAULT_DATA_PD)
        data.update(
            feature_names=feature_names,
            feature_values=feature_values,
            ice_values=ice_values,
            pd_values=pd_values,
            feature_deciles=feature_deciles,
        )
        return Explanation(meta=copy.deepcopy(self.meta), data=data)


class PartialDependence(PartialDependenceBase):
    """ Black-box implementation of partial dependence for tabular datasets.
    Supports multiple feature interactions. """

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize black-box model implementation of partial dependence.

        Parameters
        ----------
        predictor
            A prediction function which receives as input a `numpy` array of size `N x F` and outputs a
            `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input
            instances, `F` is the number of features and `T` is the number of targets.
        feature_names
            A list of feature names used for displaying results.
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
        if not callable(predictor):
            raise ValueError("The predictor must be a callable.")

        super().__init__(predictor=predictor,
                         feature_names=feature_names,
                         categorical_names=categorical_names,
                         target_names=target_names,
                         verbose=verbose)

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                features: Optional[List[Union[int, Tuple[int, int]]]] = None,
                kind: Literal['average', 'individual', 'both'] = 'average',
                percentiles: Tuple[float, float] = (0., 1.),
                grid_resolution: int = 100,
                grid_points: Optional[Dict[int, Union[List, np.ndarray]]] = None) -> Explanation:

        """
        Calculates the partial dependence for each feature and/or tuples of features with respect to the all targets
        and the reference dataset `X`.

        Parameters
        ----------
        X
            A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the
            training dataset or a representative sample.
        features
            An optional list of features or tuples of features for which to calculate the partial dependence.
            If not provided, the partial dependence will be computed for every single features in the dataset.
            Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where
            ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
        kind
            If set to ``'average'``, then only the partial dependence (PD) averaged across all samples from the dataset
            is returned. If set to ``'individual'``, then only the individual conditional expectation (ICE) is
            returned for each data point from the dataset. Otherwise, if set to ``'both'``, then both the PD and
            the ICE are returned.
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
            curves. See usage at `Partial dependence examples`_ for details

            .. _Partial dependence examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html
        """
        # kind` param sanity check.
        if kind not in Kind.__members__.values():
            raise ValueError(f"``kind='{kind}'`` is invalid. "
                             f"Accepted `kind` names are: {_get_options_string(Kind)}.")

        return super().explain(X=X,
                               features=features,
                               kind=kind,
                               percentiles=percentiles,
                               grid_resolution=grid_resolution,
                               grid_points=grid_points)

    def _compute_pd(self,
                    grid: np.ndarray,
                    features: np.ndarray,
                    X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes the partial dependence using the brute method. Code borrowed from:
        https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/inspection/_partial_dependence.py

        Parameters
        --------
        grid
            Cartesian product between feature values. Covers also the case of a single feature.
        features
            Feature column indices.
        X
             See :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.

        Returns
        -------
        Partial dependence for the given features.
        """
        predictions = []
        averaged_predictions = []
        X_eval = X.copy()

        for grid_values in grid:
            X_eval[:, features] = grid_values

            # Note: predictions is of shape
            # (n_points,) for non-multioutput regressors
            # (n_points, n_tasks) for multioutput regressors
            # (n_points, 1) for the regressors in cross_decomposition (I think)
            # (n_points, 2) for binary classification
            # (n_points, n_classes) for multiclass classification
            pred = self.predictor(X_eval)
            predictions.append(pred)

            # average over samples
            averaged_predictions.append(np.mean(pred, axis=0))

        # cast to `np.ndarray` and transpose
        predictions = np.array(predictions).T  # type: ignore[assignment]
        averaged_predictions = np.array(averaged_predictions).T  # type: ignore[assignment]
        return averaged_predictions, predictions  # type: ignore[return-value]


class TreePartialDependence(PartialDependenceBase):
    """ Tree-based model `sklearn`  implementation of the partial dependence for tabular datasets.
    Supports multiple feature interactions. This method is faster than the general black-box implementation
    but is only supported by some tree-based estimators. The computation is based on a weighted tree traversal.
    For more details on the computation, check the `sklearn documentation page`_. The supported `sklearn`
    models are: `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`,
    `HistGradientBoostingRegressor`, `HistGradientBoostingRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`.

    .. _sklearn documentation page:
            https://scikit-learn.org/stable/modules/partial_dependence.html#computation-methods
    """
    def __init__(self,
                 predictor: BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize tree-based model `sklearn` implementation of partial dependence.

        Parameters
        ----------
        predictor
            A tree-based `sklearn` estimator.
        feature_names
            A list of feature names used for displaying results.
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
        The length of the `target_names` should match the number of columns returned by a call to the
        `predictor.decision_function`. In the case of a binary classifier, the decision score consists
        of a single column. Thus, the length of the `target_names` should be one.
        """
        super().__init__(predictor=predictor,
                         feature_names=feature_names,
                         categorical_names=categorical_names,
                         target_names=target_names,
                         verbose=verbose)

        # perform sanity checks on the `sklearn` predictor
        self._sanity_check()

    def _sanity_check(self):
        """ Model sanity checks. """
        if not isinstance(self.predictor, BaseEstimator):
            raise ValueError('`TreePartialDependence` only supports `sklearn` models. '
                             'Try using the `PartialDependence` black-box alternative.')

        check_is_fitted(self.predictor)

        if not (is_classifier(self.predictor) or is_regressor(self.predictor)):
            raise ValueError('The predictor must be a fitted regressor or a fitted classifier.')

        if is_classifier(self.predictor) and isinstance(self.predictor.classes_[0], np.ndarray):
            raise ValueError('Multiclass-multioutput predictors are not supported.')

        if not isinstance(self.predictor, (BaseGradientBoosting,
                                           BaseHistGradientBoosting,
                                           DecisionTreeRegressor,
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
            raise ValueError(f'`TreePartialDependence` only supports by the following estimators: '
                             f'{supported_classes_recursion}. Try using the `PartialDependence` black-box alternative.')

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                features: Optional[List[Union[int, Tuple[int, int]]]] = None,
                percentiles: Tuple[float, float] = (0., 1.),
                grid_resolution: int = 100,
                grid_points: Optional[Dict[int, Union[List, np.ndarray]]] = None) -> Explanation:
        """
        Calculates the partial dependence for each feature and/or tuples of features with respect to the all targets
        and the reference dataset `X`.

        Parameters
        ----------
        X
            A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the
            training dataset or a representative sample.
        features
            An optional list of features or tuples of features for which to calculate the partial dependence.
            If not provided, the partial dependence will be computed for every single features in the dataset.
            Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where
            ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
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
        """
        return super().explain(X=X,
                               features=features,
                               kind=Kind.AVERAGE.value,  # only `'average'` is supported for `'recursion'` method.
                               percentiles=percentiles,
                               grid_resolution=grid_resolution,
                               grid_points=grid_points)

    def _compute_pd(self,  # type: ignore[override]
                    grid: np.ndarray,
                    features: np.ndarray,
                    **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes the PD.

        Parameters
        ----------
        grid
            Cartesian product between feature values. Covers also the case of a single feature.
        features
            Feature column indices.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Tuple consisting of the PD and ``None``.
        """
        avg_preds = self.predictor._compute_partial_dependence_recursion(grid, features)  # type: ignore[union-attr]
        return avg_preds, None


# No type check due to the generic explanation object
@no_type_check
def plot_pd(exp: Explanation,
            features: Union[List[int], Literal['all']] = 'all',
            target: Union[str, int] = 0,
            n_cols: int = 3,
            n_ice: Union[Literal['all'], int, List[int]] = 100,
            center: bool = False,
            pd_limits: Optional[Tuple[float, float]] = None,
            levels: int = 8,
            ax: Optional[Union['plt.Axes', np.ndarray]] = None,
            sharey: Optional[Literal['all', 'row']] = 'all',
            pd_num_kw: Optional[dict] = None,
            ice_num_kw: Optional[dict] = None,
            pd_cat_kw: Optional[dict] = None,
            ice_cat_kw: Optional[dict] = None,
            pd_num_num_kw: Optional[dict] = None,
            pd_num_cat_kw: Optional[dict] = None,
            pd_cat_cat_kw: Optional[dict] = None,
            fig_kw: Optional[dict] = None) -> 'np.ndarray':
    """
    Plot partial dependence curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method.
    features
        A list of features entries in the `exp.data['feature_names']` to plot the partial dependence curves for,
        or ``'all'`` to plot all the explained feature or tuples of features. This includes tuples of features.
        For example, if ``exp.data['feature_names'] = ['temp', 'hum', ('temp', 'windspeed')]`` and we want to plot
        the partial dependence only for the ``'temp'`` and ``('temp', 'windspeed')``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    target
        The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers
        denoting target index or strings denoting entries in `exp.meta['params']['target_names']`.
    n_cols
        Number of columns to organize the resulting plot into.
    n_ice
        Number of ICE plots to be displayed. Can be

         - a string taking the value ``'all'`` to display the ICE curves for every instance in the reference dataset.

         - an integer for which `n_ice` instances from the reference dataset will be sampled uniformly at random to \
         display their ICE curves.

         - a list of integers, where each integer represents an index of an instance in the reference dataset to \
         display their ICE curves.

    center
        Boolean flag to center the individual conditional expectation (ICE) curves. As mentioned in
        `Goldstein et al. (2014)`_, the heterogeneity in the model can be difficult to discern when the intercepts
        of the ICE curves cover a wide range. Centering the ICE curves removes the level effects and helps
        to visualise the heterogeneous effect.

        .. _Goldstein et al. (2014):
                https://arxiv.org/abs/1309.6392

    pd_limits
        Minimum and maximum y-limits for all the one-way PD plots. If ``None`` will be automatically inferred.
    levels
        Number of levels in the contour plot.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    sharey
        A parameter specifying whether the y-axis of the PD and ICE curves should be on the same scale
        for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``.
    pd_num_kw
        Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a
        numerical feature.
    ice_num_kw
        Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the ICE for a
        numerical feature.
    pd_cat_kw
        Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a
        categorical feature.
    ice_cat_kw
        Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the ICE for a
        categorical feature.
    pd_num_num_kw
        Keyword arguments passed to the `matplotlib.pyplot.contourf`_ function when plotting the PD for two
        numerical features.
    pd_num_cat_kw
        Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a numerical and a
        categorical feature.
    pd_cat_cat_kw
        Keyword arguments passed to the :py:meth:`alibi.utils.visualization.heatmap` functon when plotting the PD for
        two categorical features.
    fig_kw
        Keyword arguments passed to the `matplotlib.figure.set`_ function.

        .. _matplotlib.pyplot.plot:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

        .. _matplotlib.pyplot.contourf:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    -------
    An array of `plt.Axes` with the resulting partial dependence plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}
    fig_kw = {**default_fig_kw, **fig_kw}

    if features == 'all':
        features = range(0, len(exp.data['feature_names']))
    else:
        for ifeatures in features:
            if ifeatures >= len(exp.data['feature_names']):
                raise IndexError(f"The `features` indices must be less than the "
                                 f"``len(feature_names) = {len(exp.data['feature_names'])}``. "
                                 f"Received {ifeatures}.")

    # set target index
    if isinstance(target, str):
        try:
            target_idx = exp.meta['params']['target_names'].index(target)
        except ValueError:
            raise ValueError(f"Unknown `target` name. Received {target}. "
                             f"Available values are: {exp.meta['params']['target_names']}.")
    else:
        target_idx = target
        if target_idx >= len(exp.meta['params']['target_names']):
            raise IndexError(f"Target index out of range. Received {target_idx}. "
                             f"The number of targets is {len(exp.meta['params']['target_names'])}.")

    # corresponds to the number of subplots
    n_features = len(features)

    # create axes
    if ax is None:
        fig, ax = plt.subplots()

    def _is_categorical(feature):
        feature_idx = exp.meta['params']['feature_names'].index(feature)
        return feature_idx in exp.meta['params']['categorical_names']

    if isinstance(ax, plt.Axes) and n_features != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_features)
        n_rows = math.ceil(n_features / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)
        for i, spec in enumerate(list(gs)[:n_features]):
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_features:
            raise ValueError(f"Expected ax to have {n_features} axes, got {ax.size}")
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    # create plots
    one_way_axs = {}

    for i, (ifeatures, ax_ravel) in enumerate(zip(features, axes_ravel)):
        # extract the feature names
        feature_names = exp.data['feature_names'][ifeatures]

        # if it is tuple, then we need a 2D plot and address 4 cases: (num, num), (num, cat), (cat, num), (cat, cat)
        if isinstance(feature_names, tuple):
            f0, f1 = feature_names

            if (not _is_categorical(f0)) and (not _is_categorical(f1)):
                ax, ax_pd_limits = _plot_two_pd_num_num(exp=exp,
                                                        feature=ifeatures,
                                                        target_idx=target_idx,
                                                        levels=levels,
                                                        ax=ax_ravel,
                                                        pd_num_num_kw=pd_num_num_kw)

            elif _is_categorical(f0) and _is_categorical(f1):
                ax, ax_pd_limits = _plot_two_pd_cat_cat(exp=exp,
                                                        feature=ifeatures,
                                                        target_idx=target_idx,
                                                        ax=ax_ravel,
                                                        pd_cat_cat_kw=pd_cat_cat_kw)

            else:
                ax, ax_pd_limits = _plot_two_pd_num_cat(exp=exp,
                                                        feature=ifeatures,
                                                        target_idx=target_idx,
                                                        pd_limits=pd_limits,
                                                        ax=ax_ravel,
                                                        pd_num_cat_kw=pd_num_cat_kw)

        else:
            if _is_categorical(feature_names):
                ax, ax_pd_limits = _plot_one_pd_cat(exp=exp,
                                                    feature=ifeatures,
                                                    target_idx=target_idx,
                                                    center=center,
                                                    pd_limits=pd_limits,
                                                    n_ice=n_ice,
                                                    ax=ax_ravel,
                                                    pd_cat_kw=pd_cat_kw,
                                                    ice_cat_kw=ice_cat_kw)
            else:
                ax, ax_pd_limits = _plot_one_pd_num(exp=exp,
                                                    feature=ifeatures,
                                                    target_idx=target_idx,
                                                    center=center,
                                                    pd_limits=pd_limits,
                                                    n_ice=n_ice,
                                                    ax=ax_ravel,
                                                    pd_num_kw=pd_num_kw,
                                                    ice_num_kw=ice_num_kw)

        # group the `ax_ravel` that share the appropriate y axes.
        if ax_pd_limits is not None:
            if sharey == 'all':
                if one_way_axs.get('all', None) is None:
                    one_way_axs['all'] = []

                # add them all in the same group
                one_way_axs['all'].append((ax, ax_pd_limits))
            elif sharey == 'row':
                # identify the row to which they belong
                row = i // n_cols

                if one_way_axs.get(row, None) is None:
                    one_way_axs[row] = []

                # add them the `row` group
                one_way_axs[row].append((ax, ax_pd_limits))
            else:
                # if no axis are share, each `ax_ravel` will have its own group
                one_way_axs[i] = [(ax, ax_pd_limits)]

    #  share the y-axis for the axes within the same group and set the `ymin`, `ymax` values.
    #  This step is necessary and applied here because `vlines` overwrites the `ylim`.
    for ax_group in one_way_axs.values():
        min_val = min([ax_pd_lim[0] for _, ax_pd_lim in ax_group])
        max_val = max([ax_pd_lim[1] for _, ax_pd_lim in ax_group])
        ax_group[0][0].get_shared_y_axes().join(ax_group[0][0], *[ax[0] for ax in ax_group[1:]])
        ax_group[0][0].set_ylim(min_val, max_val)

    fig.set(**fig_kw)
    return axes


def _sample_ice(ice_values: np.ndarray, n_ice: Union[Literal['all'], int, List[int]]) -> np.ndarray:
    """
    Samples `ice_values` based on the `n_ice` argument.

    Parameters
    ----------
    ice_values
        Array of ice_values of dimension `V x N`, where `V` is the number of feature values where the PD is computed,
        and `N` is the number of instances in the reference dataset.
    n_ice
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd`.
    """
    if n_ice == 'all':
        return ice_values

    _, N = ice_values.shape
    if isinstance(n_ice, numbers.Integral):
        if n_ice >= N:  # type: ignore[operator]
            logger.warning('`n_ice` is greater than the number of instances in the reference dataset. '
                           'Automatically setting `n_ice` to the number of instances in the reference dataset.')
            return ice_values

        if n_ice <= 0:  # type: ignore[operator]
            raise ValueError('`n_ice` must be an integer grater than 0.')

        indices = np.random.choice(a=N, size=n_ice, replace=False)
        return ice_values[:, indices]

    if isinstance(n_ice, list):
        n_ice = np.unique(n_ice)  # type: ignore[assignment]
        if not np.all(n_ice < N) or not np.all(n_ice >= 0):  # type: ignore[operator]
            raise ValueError(f'Some indices in `n_ice` are out of bounds. Ensure that all indices are '
                             f'greater or equal than 0 and less than {N}.')
        return ice_values[:, n_ice]

    raise ValueError(f"Unknown `n_ice` values. `n_ice` can be a string taking value 'all', "
                     f"an integer, or a list of integers. Received {n_ice}.")


def _process_pd_ice(exp: Explanation,
                    pd_values: Optional[np.ndarray] = None,
                    ice_values: Optional[np.ndarray] = None,
                    n_ice: Union[Literal['all'], int, List[int]] = 'all',
                    center: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process the `pd_values` and `ice_values` before plotting. Centers the plots if necessary and samples
    the `ice_values` for visualization purposes.

    Parameters
    ----------
    exp, n_ice, center
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    pd_values
        Array of ice_values of dimension `V` (i.e. `(V, )`), where V is the number of feature values where
        the PD is computed.
    ice_values
        Array of ice_values of dimension `V x N`, where `V` is the number of feature values where the PD is computed,
        and `N` is the number of instances in the reference dataset.

    Returns
    -------
    Tuple containing the processed `pd_values` and `ice_values`.
    """
    # pdp processing
    if exp.meta['params']['kind'] == Kind.BOTH and center:
        pd_values = pd_values - pd_values[0]  # type: ignore[index]

    # ice processing
    if exp.meta['params']['kind'] in [Kind.INDIVIDUAL, Kind.BOTH]:
        # sample ice values for visualization purposes
        ice_values = _sample_ice(ice_values=ice_values, n_ice=n_ice)  # type: ignore[arg-type]

        # center ice values if necessary
        if center:
            ice_values = ice_values - ice_values[0:1, :]

    return pd_values, ice_values


# No type check due to the generic explanation object
@no_type_check
def _plot_one_pd_num(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     center: bool = False,
                     pd_limits: Optional[Tuple[float, float]] = None,
                     n_ice: Union[Literal['all'], int, List[int]] = 100,
                     ax: Optional['plt.Axes'] = None,
                     pd_num_kw: Optional[dict] = None,
                     ice_num_kw: Optional[dict] = None) -> Tuple['plt.Axes', Optional[Tuple[float, float]]]:
    """
    Plots one way partial dependence curve for a single numerical feature.

    Parameters
    ----------
    exp, feature, center, pd_limits, n_ice, pd_num_kw, ice_num_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    target_idx
        The target index for which to plot the partial dependence (PD) curves. An integer
        denoting target index in `exp.meta['params]['target_names']`
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.

    Returns
    -------
    `matplotlib` axes and a tuple containing the minimum and maximum y-limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    feature_values = exp.data['feature_values'][feature]
    pd_values = exp.data['pd_values'][feature][target_idx] if (exp.data['pd_values'] is not None) else None
    ice_values = exp.data['ice_values'][feature][target_idx].T if (exp.data['ice_values'] is not None) else None

    # process `pd_values` and `ice_values`
    pd_values, ice_values = _process_pd_ice(exp=exp,
                                            pd_values=pd_values,
                                            ice_values=ice_values,
                                            n_ice=n_ice,
                                            center=center)

    if exp.meta['params']['kind'] == Kind.AVERAGE:
        default_pd_num_kw = {'markersize': 2, 'marker': 'o', 'label': None}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}
        ax.plot(feature_values, pd_values, **pd_num_kw)

    elif exp.meta['params']['kind'] == Kind.INDIVIDUAL:
        default_ice_graph_kw = {'color': 'lightsteelblue', 'label': None}
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}
        ax.plot(feature_values, ice_values, **ice_num_kw)

    else:
        default_pd_num_kw = {'linestyle': '--', 'linewidth': 2, 'color': 'tab:orange', 'label': 'average'}
        pd_num_kw = default_pd_num_kw if pd_num_kw is None else {**default_pd_num_kw, **pd_num_kw}

        default_ice_graph_kw = {'alpha': 0.6, 'color': 'lightsteelblue', 'label': None}
        ice_num_kw = default_ice_graph_kw if ice_num_kw is None else {**default_ice_graph_kw, **ice_num_kw}

        ax.plot(feature_values, ice_values, **ice_num_kw)
        ax.plot(feature_values, pd_values, **pd_num_kw)
        ax.legend()

    # save the `ylim` as they will be overwritten by `ax.vlines`
    ylim = ax.get_ylim() if pd_limits is None else pd_limits

    # add deciles markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.data['feature_deciles'][feature][1:-1], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.data['feature_names'][feature])
    ax.set_ylabel(exp.meta['params']['target_names'][target_idx])
    return ax, ylim


# No type check due to the generic explanation object
@no_type_check
def _plot_one_pd_cat(exp: Explanation,
                     feature: int,
                     target_idx: int,
                     pd_limits: Optional[Tuple[float, float]] = None,
                     center: bool = False,
                     n_ice: Union[Literal['all'], int, List[str]] = 100,
                     ax: Optional['plt.Axes'] = None,
                     pd_cat_kw: Optional[dict] = None,
                     ice_cat_kw: Optional[dict] = None) -> Tuple['plt.Axes', Optional[Tuple[float, float]]]:
    """
    Plots one way partial dependence curve for a single categorical feature.

    Parameters
    ----------
    exp, feature, center, pd_limits, n_ice, pd_cat_kw, ice_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    target_idx
        The target index for which to plot the partial dependence (PD) curves. An integer
        denoting target index in `exp.meta['params'].target_names`
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.

    Returns
    -------
    `matplotlib` axes and a tuple containing the minimum and maximum y-limits.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    feature_names = exp.data['feature_names'][feature]
    feature_values = exp.data['feature_values'][feature]
    pd_values = exp.data['pd_values'][feature][target_idx] if (exp.data['pd_values'] is not None) else None
    ice_values = exp.data['ice_values'][feature][target_idx].T if (exp.data['ice_values'] is not None) else None

    # process `pd_values` and `ice_values`
    pd_values, ice_values = _process_pd_ice(exp=exp,
                                            pd_values=pd_values,
                                            ice_values=ice_values,
                                            n_ice=n_ice,
                                            center=center)

    feature_index = exp.meta['params']['feature_names'].index(feature_names)
    labels = [exp.meta['params']['categorical_names'][feature_index][i] for i in feature_values.astype(np.int32)]

    if exp.meta['params']['kind'] == Kind.AVERAGE:
        default_pd_graph_kw = {'markersize': 8, 'marker': 's', 'color': 'tab:blue'}
        pd_cat_kw = default_pd_graph_kw if pd_cat_kw is None else {**default_pd_graph_kw, **pd_cat_kw}
        ax.plot(labels, pd_values, **pd_cat_kw)

    elif exp.meta['params']['kind'] == Kind.INDIVIDUAL:
        default_ice_cat_kw = {'markersize': 4, 'marker': 's', 'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}
        ax.plot(labels, ice_values, **ice_cat_kw)

    else:
        default_pd_cat_kw = {'markersize': 8, 'marker': 's', 'color': 'tab:orange', 'label': 'average'}
        pd_cat_kw = default_pd_cat_kw if pd_cat_kw is None else {**default_pd_cat_kw, **pd_cat_kw}

        default_ice_cat_kw = {'alpha': 0.6, 'markersize': 4, 'marker': 's', 'color': 'lightsteelblue'}
        ice_cat_kw = default_ice_cat_kw if ice_cat_kw is None else {**default_ice_cat_kw, **ice_cat_kw}

        ax.plot(labels, ice_values, **ice_cat_kw)
        ax.plot(labels, pd_values, **pd_cat_kw)
        ax.legend()

    # save `ylim`
    ylim = ax.get_ylim() if pd_limits is None else pd_limits

    # rotate xticks labels
    ax.tick_params(axis='x', rotation=90)

    # set axis labels
    ax.set_xlabel(feature_names)
    ax.set_ylabel(exp.meta['params']['target_names'][target_idx])
    return ax, ylim


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_num_num(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         levels: int = 8,
                         ax: Optional['plt.Axes'] = None,
                         pd_num_num_kw: Optional[dict] = None) -> Tuple['plt.Axes', Optional[Tuple[float, float]]]:
    """
    Plots two ways partial dependence curve for two numerical features.

    Parameters
    ----------
    exp, feature, pd_num_num_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    target_idx
        The target index for which to plot the partial dependence (PD) curves. An integer
        denoting target index in `exp.meta['params']['target_names']`
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.

    Returns
    -------
    `matplotlib` axes and ``None``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if exp.meta['params']['kind'] not in [Kind.AVERAGE, Kind.BOTH]:
        raise ValueError("Can only plot partial dependence for `kind` in `['average', 'both']`.")

    if ax is None:
        ax = plt.gca()

    # set contour plot default params
    default_pd_num_num_kw = {"alpha": 0.75}
    pd_num_num_kw = default_pd_num_num_kw if pd_num_num_kw is None else {**default_pd_num_num_kw, **pd_num_num_kw}

    feature_values = exp.data['feature_values'][feature]
    pd_values = exp.data['pd_values'][feature][target_idx]

    X, Y = np.meshgrid(feature_values[0], feature_values[1])
    Z, Z_min, Z_max = pd_values.T, pd_values.min(), pd_values.max()

    if Z_max > Z_min:
        Z_level = np.linspace(Z_min, Z_max, levels)
    else:
        # this covers the case when `Z_min` equals `Z_max`, for which `Z_level` will be constant.
        # Note that `ax.contourf` accepts only increasing `Z_levels`, otherwise it throws an error.
        Z_level, Z_min, Z_max = None, None, None

    CS = ax.contour(X, Y, Z, levels=Z_level, linewidths=0.5, colors="k")
    ax.contourf(X, Y, Z, levels=Z_level, vmax=Z_max, vmin=Z_min, **pd_num_num_kw)
    ax.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)

    # create the deciles line for the vertical & horizontal axis
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # the horizontal lines do not display (same for the sklearn)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.data['feature_deciles'][feature][0][1:-1], 0, 0.05, transform=trans)
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.hlines(exp.data['feature_deciles'][feature][1][1:-1], 0, 0.05, transform=trans)

    # reset xlim and ylim since they are overwritten by hlines and vlines
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # set x & y labels
    ax.set_xlabel(exp.data['feature_names'][feature][0])
    ax.set_ylabel(exp.data['feature_names'][feature][1])
    return ax, None


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_num_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         pd_limits: Optional[Tuple[float, float]] = None,
                         ax: Optional['plt.Axes'] = None,
                         pd_num_cat_kw: Optional[dict] = None) -> Tuple['plt.Axes', Optional[Tuple[float, float]]]:
    """
    Plots two ways partial dependence curve for a numerical feature and a categorical feature.

    Parameters
    ----------
    exp, feature, pd_num_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    target_idx
        The target index for which to plot the partial dependence (PD) curves. An integer
        denoting target index in `exp.meta['params']['target_names'].`
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.

    Returns
    -------
    `matplotlib` axes and a tuple containing the minimum and maximum y-limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if exp.meta['params']['kind'] not in [Kind.AVERAGE, Kind.BOTH]:
        raise ValueError("Can only plot partial dependence for `kind` in `['average', 'both']`.")

    if ax is None:
        ax = plt.gca()

    def _is_categorical(feature):
        feature_idx = exp.meta['params']['feature_names'].index(feature)
        return feature_idx in exp.meta['params']['categorical_names']

    # extract feature values and partial dependence values
    feature_values = exp.data['feature_values'][feature]
    feature_deciles = exp.data['feature_deciles'][feature]
    pd_values = exp.data['pd_values'][feature][target_idx]

    # find which feature is categorical and which one is numerical
    feature_names = exp.data['feature_names'][feature]
    if _is_categorical(feature_names[0]):
        feature_names = feature_names[::-1]
        feature_values = feature_values[::-1]
        feature_deciles = feature_deciles[::-1]
        pd_values = pd_values.T

    # define labels
    cat_feature_index = exp.meta['params']['feature_names'].index(feature_names[1])
    labels = [exp.meta['params']['categorical_names'][cat_feature_index][i]
              for i in feature_values[1].astype(np.int32)]

    # plot lines
    default_pd_num_cat_kw = {'markersize': 2, 'marker': 'o'}
    pd_num_cat_kw = default_pd_num_cat_kw if pd_num_cat_kw is None else {**default_pd_num_cat_kw, **pd_num_cat_kw}
    ax.plot([], [], ' ', label=feature_names[1])

    for i in range(pd_values.shape[1]):
        x, y = feature_values[0], pd_values[:, i]
        pd_num_cat_kw.update({'label': labels[i]})
        ax.plot(x, y, **pd_num_cat_kw)

    # save `ylim` as they will be overwritten by `ax.vlines`
    ylim = ax.get_ylim() if pd_limits is None else pd_limits

    # add deciles markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(feature_deciles[0][1:-1], 0, 0.05, transform=trans)

    ax.set_ylabel(exp.meta['params']['target_names'][target_idx])
    ax.set_xlabel(feature_names[0])
    ax.legend()
    return ax, ylim


# No type check due to the generic explanation object
@no_type_check
def _plot_two_pd_cat_cat(exp: Explanation,
                         feature: int,
                         target_idx: int,
                         ax: Optional['plt.Axes'] = None,
                         pd_cat_cat_kw: Optional[dict] = None) -> Tuple['plt.Axes', Optional[Tuple[float, float]]]:
    """
    Plots two ways partial dependence curve for two categorical features.

    Parameters
    ----------
    exp, feature, pd_cat_cat_kw
        See :py:meth:`alibi.explainers.partial_dependence.plot_pd` method.
    target_idx
        The target index for which to plot the partial dependence (PD) curves. An integer
        denoting target index in `exp.meta['params']['target_names']`.
    ax
        Pre-existing axes for the plot. Otherwise, call `matplotlib.pyplot.gca()` internally.

    Return
    ------
    `matplotlib` axes and ``None``.
    """
    import matplotlib.pyplot as plt

    from alibi.utils.visualization import heatmap

    if ax is None:
        ax = plt.gca()

    if exp.meta['params']['kind'] not in [Kind.AVERAGE, Kind.BOTH]:
        raise ValueError("Can only plot partial dependence for `kind` in `['average', 'both']`.")

    feature_names = exp.data['feature_names'][feature]
    feature_values = exp.data['feature_values'][feature]
    pd_values = exp.data['pd_values'][feature][target_idx]

    # extract labels for each categorical features
    feature0_index = exp.meta['params']['feature_names'].index(feature_names[0])
    feature1_index = exp.meta['params']['feature_names'].index(feature_names[1])
    labels0 = [exp.meta['params']['categorical_names'][feature0_index][i]
               for i in feature_values[0].astype(np.int32)]
    labels1 = [exp.meta['params']['categorical_names'][feature1_index][i]
               for i in feature_values[1].astype(np.int32)]

    # plot heatmap
    default_pd_cat_cat_kw = {
        'annot': True,
        'fmt': '{x:.2f}',
        'linewidths': 1.5,
        'yticklabels': labels0,
        'xticklabels': labels1,
        'aspect': 'auto'
    }
    pd_cat_cat_kw = default_pd_cat_cat_kw if pd_cat_cat_kw is None else {**default_pd_cat_cat_kw, **pd_cat_cat_kw}
    heatmap(pd_values, ax=ax, **pd_cat_cat_kw)

    # set ticks labels
    ax.set_xticklabels(labels1)
    ax.set_yticklabels(labels0)

    # set axis labels
    ax.set_xlabel(exp.data['feature_names'][feature][1])
    ax.set_ylabel(exp.data['feature_names'][feature][0])
    return ax, None
