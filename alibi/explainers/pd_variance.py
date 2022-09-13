import copy
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Dict, Union, Tuple

from alibi.api.defaults import DEFAULT_META_PDVARIANCE, DEFAULT_DATA_PDVARIANCE
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.partial_dependence import Kind, PartialDependence, TreePartialDependence
from sklearn.base import BaseEstimator


logger = logging.getLogger(__name__)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


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
            provided in `features`.
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
                https://docs.seldon.io/projects/alibi/en/stable/methods/VariancePartialDependence.html
        """

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

        if method == 'importance':
            # compute feature importance
            buffers = self._compute_feature_importance(pd_explanation=pd_explanation, features=features)
        elif method == 'interaction':
            # compute feature interaction
            buffers = self._compute_feature_interaction(pd_explanation=pd_explanation, features=features)
        else:
            raise ValueError(f"Unknown mode. Received ``method={method}``. "
                             f"Supported values: 'importance' | 'interaction'")

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
            List of PD values for each feature in `features`. Each PD value is an array with the shape
            `T x N1 ... x Nk` where `T` is the number of targets and `Ni` is the number of feature values
            along the `i` axis.

        Returns
        -------
        An array of size `F x T x N1 x ... N(k-1)`, where `F` is the number of explained features, `T` is the number
        of targets, and `Ni` is the number of feature values along the `i` axis.
        """
        feature_variance = []

        for pd_values, f in zip(pd_values, features):
            # `pd_values` is a tensor of size `T x N1 x ... Nk`, where `T` is the number
            # of targets and `Ni` is the number of feature values along the axis `i`
            ft_var = self._compute_pd_variance_cat(pd_values) if (f in self.pd_explainer.categorical_names) \
                else self._compute_pd_variance_num(pd_values)
            feature_variance.append(ft_var[None, ...])

        # stack the feature importance such that the array has the shape `F x T x N1 ... N(k-1)`
        return np.concatenate(feature_variance, axis=0)  # type: ignore[assignment]

    def _compute_pd_variance_num(self, pd_values: np.ndarray) -> np.ndarray:
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

    def _compute_pd_variance_cat(self, pd_values: np.ndarray) -> np.array:
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

    def _compute_feature_importance(self, pd_explanation: Explanation, features: List[int]):
        return {
            'feature_deciles': pd_explanation.data['feature_deciles'],
            'pd_values': pd_explanation.data['pd_values'],
            'feature_values': pd_explanation.data['feature_values'],
            'feature_names': pd_explanation.data['feature_names'],
            'feature_importance': self._compute_pd_variance(features=features,
                                                            pd_values=pd_explanation.data['pd_values']).T,
        }

    def _compute_feature_interaction(self, features: List[Tuple[int, int]], pd_explanation: Explanation):
        buffers = {
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
            TODO

        Returns
        -------
        `Explanation` object.
        """
        data = copy.deepcopy(DEFAULT_DATA_PDVARIANCE)
        data.update(feature_deciles=buffers['feature_deciles'],
                    pd_values=buffers['pd_values'],
                    feature_values=buffers['feature_values'],
                    feature_names=buffers['feature_names'])

        if self.meta['params']['method'] == 'importance':
            data.update(feature_importance=buffers['feature_importance'])
        else:
            data.update(feature_interaction=buffers['feature_interaction'])

        return Explanation(meta=self.meta, data=data)


def plot_feature_importance(exp: Explanation,
                            features: Union[List[int], Literal['all']] = 'all',
                            targets: Union[List[Union[str, int]], Literal['all']] = 'all',
                            ax: Optional[Union['plt.Axes', np.ndarray]] = None,
                            sort: bool = True,
                            **kwargs) -> 'plt.Axes':
    """
    Horizontal bar plot for feature importance.

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
    targets
        The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers
        denoting target index or strings denoting entries in `exp.meta['params']['target_names']`.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    **kwargs
        Keyword arguments passed to the `pandas.DataFrame.plot.barh`_ function.

        .. _pandas.DataFrame.plot.barh:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.barh.html

    Returns
    -------
    `plt.Axes` with the resulting feature importance plot.
    """

    if features == 'all':
        feature_indices = list(range(len(exp.data['feature_names'])))
        feature_names = exp.data['feature_names']
    else:
        feature_indices, feature_names = features, []
        for ifeatures in features:
            if ifeatures > len(exp.data['feature_names']):
                raise ValueError(f"The `features` indices must be less than the "
                                 f"``len(feature_names) = {len(exp.data['feature_names'])}``. "
                                 f"Received {ifeatures}.")
            else:
                feature_names.append(exp.data['feature_names'][ifeatures])

    # set target indices
    if targets == 'all':
        target_indices = list(range(len(exp.meta['params']['target_names'])))
        target_names = exp.meta['params']['target_names']
    else:
        target_indices, target_names = [], []

        for target in targets:
            if isinstance(target, str):
                try:
                    target_idx = exp.meta['params']['target_names'].index(target)
                    target_name = target
                except ValueError:
                    raise ValueError(f"Unknown `target` name. Received {target}. "
                                     f"Available values are: {exp.meta['params']['target_names']}.")
            else:
                try:
                    target_idx = target
                    target_name = exp.meta['params']['target_names'][target]
                except ValueError:
                    raise IndexError(f"Target index out of range. Received {target}. "
                                     f"The number of targets is {len(exp.meta['params']['target_names'])}.")
            target_indices.append(target_idx)
            target_names.append(target_name)

    # create `pandas.DataFrame` for easy display
    data = exp.data['feature_importance'][target_indices].T[feature_indices]

    if sort:
        if len(target_indices) > 1:
            logging.warning('Cannot sort features by importance when multiple targets are displayed on the same plot.')
        else:
            sorted_indices = np.argsort(data, axis=0).reshape(-1)
            data = data[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]

    df = pd.DataFrame(data=data, columns=target_names, index=feature_names)
    return df.plot.barh(ax=ax, **kwargs)
