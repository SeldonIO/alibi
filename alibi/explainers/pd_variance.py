import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Dict, Union, Tuple

from alibi.api.defaults import DEFAULT_META_PDVARIANCE, DEFAULT_DATA_PDVARIANCE
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.partial_dependence import Kind, PartialDependence


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class PartialDependenceVariance(Explainer):
    """ Black-box implementation of the variance partial dependence feature importance for tabular datasets. """

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 target_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize black-box model implementation for the variance partial dependence feature importance.

        Parameters
        ----------
        predictor
            A prediction function which receives as input a `numpy` array of size `N x F` and outputs a
            `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input
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
        self.pd_explainer = PartialDependence(predictor=predictor,
                                              feature_names=feature_names,
                                              categorical_names=categorical_names,
                                              target_names=target_names,
                                              verbose=verbose)

    def explain(self,
                X: np.ndarray,
                features: Optional[List[int]] = None,
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

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated partial dependence
            curves. See usage at `Partial dependence examples`_ for details

            .. _Partial dependence examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/VariancePartialDependence.html
        """
        # compute partial dependence functions
        pd_explanation = self.pd_explainer.explain(X=X,
                                                   features=features,  # type: ignore[arg-type]
                                                   kind=Kind.AVERAGE.value,
                                                   percentiles=percentiles,
                                                   grid_resolution=grid_resolution,
                                                   grid_points=grid_points)

        # filter explained numerical and categorical features
        features = [pd_explanation.meta['params']['feature_names'].index(f)
                    for f in pd_explanation.data['feature_names']]

        # compute feature importance
        feature_importance = []

        for pd_values, f in zip(pd_explanation.data['pd_values'], features):
            # `pd_values` is a tensor of size `T x N`, where `T` is the number
            # of targets and `N` is the number of feature values

            if f in self.pd_explainer.categorical_names:   # type: ignore[operator]
                # compute feature importance for categorical features
                ft_imp = (pd_values.max(axis=-1) - pd_values.min(axis=-1)) / 4
            else:
                # compute feature importance for numerical features
                ft_imp = np.std(pd_values, axis=-1, ddof=1)

            feature_importance.append(ft_imp.reshape(-1, 1))

        # stack the feature importance such that the array has the shape `T x F`, where ``T` is the
        # number of targets and `F` is the number of explained features
        feature_importance = np.hstack(feature_importance)  # type: ignore[assignment]

        # update meta and return built explanation
        self.meta['params'].update(self.pd_explainer.meta['params'])
        return self._build_explanation(pd_explanation=pd_explanation,
                                       feature_importance=feature_importance)  # type: ignore[arg-type]

    def _build_explanation(self, pd_explanation: Explanation, feature_importance: np.ndarray) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        pd_explanation
            The `PartialDependence` explanation object.
        feature_importance
            Array of feature importance of size `T x F`, where `T` is the number of targets and `F` is the number
            of features to be explained.

        Returns
        -------
        `Explanation` object.
        """

        data = copy.deepcopy(DEFAULT_DATA_PDVARIANCE)
        data.update(feature_deciles=pd_explanation.data['feature_deciles'],
                    pd_values=pd_explanation.data['pd_values'],
                    feature_values=pd_explanation.data['feature_values'],
                    feature_names=pd_explanation.data['feature_names'],
                    feature_importance=feature_importance)
        return Explanation(meta=self.meta, data=data)


class TreePartialDependenceVariance(Explainer):
    pass


def plot_feature_importance(exp: Explanation,
                            features: Union[List[int], Literal['all']] = 'all',
                            targets: Union[List[Union[str, int]], Literal['all']] = 'all',
                            ax: Optional[Union['plt.Axes', np.ndarray]] = None,
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
    df = pd.DataFrame(data=data, columns=target_names, index=feature_names)
    return df.plot.barh(ax=ax, **kwargs)
