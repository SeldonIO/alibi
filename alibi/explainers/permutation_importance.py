import copy
import inspect
import logging
import math
import numbers
import sys
from collections import defaultdict
from enum import Enum
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from alibi.api.defaults import (DEFAULT_DATA_PERMUTATION_IMPORTANCE,
                                DEFAULT_META_PERMUTATION_IMPORTANCE)
from alibi.api.interfaces import Explainer, Explanation

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

logger = logging.getLogger(__name__)


class Method(str, Enum):
    """ Enumeration of supported method. """
    EXACT = 'exact'
    ESTIMATE = 'estimate'


class Kind(str, Enum):
    """ Enumeration of supported kind. """
    DIFFERENCE = 'difference'
    RATIO = 'ratio'


class PermutationImportance(Explainer):
    """ Implementation of the permutation feature importance for tabular dataset. The method measure the importance
    of a feature as the relative increase in the loss function when the feature values are permuted. Supports
    black-box models.

    For details of the method see the papers:

     - https://link.springer.com/article/10.1023/A:1010933404324
     - https://arxiv.org/abs/1801.01489
    """

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize the permutation feature importance.

        Parameters
        ----------
        predictor
            A prediction function which receives as input a `numpy` array of size `N x F`, and outputs a
            `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances,
            `F` is the number of features, and `T` is the number of targets. Note that the output shape must be
            compatible with the loss functions provided in the
            :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        feature_names
            A list of feature names used for displaying results.
        verbose
            Whether to print the progress of the explainer.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PERMUTATION_IMPORTANCE))
        self.predictor = predictor
        self.feature_names = feature_names
        self.verbose = verbose

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                y: np.ndarray,
                loss_fns: Optional[
                    Union[
                        Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ]
                ] = None,
                score_fns: Optional[
                    Union[
                        Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ]
                ] = None,
                features: Optional[List[Union[int, Tuple[int, ...]]]] = None,
                method: Literal["estimate", "exact"] = "estimate",
                kind: Literal["ratio", "difference"] = "ratio",
                n_repeats: int = 50,
                sample_weight: Optional[np.ndarray] = None) -> Explanation:
        """
        Computes the permutation feature importance for each feature with respect to the give loss functions and
        the dataset `(X, y)`.

        Parameters
        ----------
        X
            A `N x F` input feature dataset used to calculate the permutation feature importance. This is typically the
            test dataset.
        y
            A `N` (i.e. `(N, )`) ground-truth labels array corresponding the input feature `X`.
        loss_fns
            A loss function or a dictionary of loss functions having as keys the name of the loss functions and as
            values the loss functions. Note that the `predictor` output must be compatible with every loss functions.
            Every loss function is expected to receive the following arguments:

             - `y_true` : ``np.ndarray`` -  a `numpy` array of ground-truth labels.

             - `y_pred` : ``np.ndarray`` - a `numpy` array of model predictions. This corresponds to the output of
             the model

             - `sample_weight`: ``Optional[np.ndarray]`` - a `numpy` array of sample weights.

        features
            An optional list of features or tuples of features for which to calculate the partial dependence.
            If not provided, the partial dependence will be computed for every single features in the dataset.
            Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where
            ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
        method
            The method to be used to compute the feature importance. If set to ``'switch'``, a "switch" operation is
            performed across all observed pairs, by excluding pairings that are actually observed in the original
            dataset. This operation is quadratic in the number of samples (`N x (N - 1)` samples) and thus can be
            computationally intensive. If set to ``'divide'``, the dataset will be divided in half and the first
            half's values of the remaining ground-truth labels and the rest of the feature is matched with the
            second half's values of the permuted features, and the other way around. This method is computationally
            lighter and provides estimate error bars given by the standard deviation.
        kind
            Whether to report the importance as the error ratio or the error difference. Available values are:
            ``'ratio'`` | ``'difference'``.
        n_repeats
            Number of times to permute the features. Considered only when ``method='estimate'``.
        sample_weight
            Optional weight for each sample instance.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated permutation feature
            importance. See usage at `Permutation feature importance examples`_ for details

            .. _Permutation feature importance examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html
        """
        n_features = X.shape[1]

        if (loss_fns is None) and (score_fns is None):
            raise ValueError('At least one loss function or a score function must be provided.')

        # initialize loss functions dictionary
        if loss_fns is None:
            loss_fns = {}
        elif callable(loss_fns):
            loss_fns = {'loss': loss_fns}

        # initialize score functions dictionary
        if score_fns is None:
            score_fns = {}
        elif callable(score_fns):
            score_fns = {'score': score_fns}

        # set the `features_names` when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # construct `feature_names` based on the `features`. If `features` is ``None``, then initialize
        # `features` with all single feature available in the dataset.
        if features:
            feature_names = [tuple([self.feature_names[f] for f in features])
                             if isinstance(features, tuple) else self.feature_names[features]
                             for features in features]
        else:
            feature_names = self.feature_names  # type: ignore[assignment]
            features = list(range(n_features))

        # unaltered model predictions
        y_pred = self.predictor(X)

        # compute base loss
        loss_orig = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                           y_true=y,
                                                           y_pred=y_pred,
                                                           sample_weight=sample_weight)

        # compute base score
        score_orig = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                            y_true=y,
                                                            y_pred=y_pred,
                                                            sample_weight=sample_weight)

        # compute permutation feature importance for every feature
        # TODO: implement parallel version - future work as it can be done for ALE too
        individual_feature_importance = []

        for ifeatures in tqdm(features, disable=not self.verbose):
            individual_feature_importance.append(
                self._compute_permutation_importance(
                    X=X,
                    y=y,
                    loss_fns=loss_fns,
                    score_fns=score_fns,
                    method=method,
                    kind=kind,
                    n_repeats=n_repeats,
                    sample_weight=sample_weight,
                    features=ifeatures,
                    loss_orig=loss_orig,
                    score_orig=score_orig
                )
            )

        # update meta data params
        self.meta['params'].update(feature_names=feature_names,
                                   method=method,
                                   kind=kind,
                                   n_repeats=n_repeats,
                                   sample_weight=sample_weight)

        # build and return the explanation object
        return self._build_explanation(feature_names=feature_names,  # type: ignore[arg-type]
                                       individual_feature_importance=individual_feature_importance)

    @staticmethod
    def _compute_metrics(metric_fns: Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]],
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None,
                         metrics: Optional[Dict[str, List[float]]] = None):
        """
        Helper function to compute multiple metrics.

        Parameters
        ----------
        metric_fns
            A dictionary of metric functions having as keys the name of the metric functions and as
            values the metric functions.
        y_true
            Ground truth targets.
        y_pred
            Predicted outcome as returned by the classifier.
        sample_weight
            Weight of each sample instance.
        metrics
            An optional dictionary of metrics, having as keys the name of the metric and as value the evaluation of
            the metric.

        Returns
        -------
        Updated `metrics` dictionary.
        """
        if metrics is None:
            metrics = defaultdict(list)

        # compute base metric
        for metric_name, metric_fn in metric_fns.items():
            metrics[metric_name].append(
                PermutationImportance._compute_metric(metric_fn=metric_fn,
                                                      y_true=y_true,
                                                      y_pred=y_pred,
                                                      sample_weight=sample_weight)
            )
        return metrics

    @staticmethod
    def _compute_metric(metric_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Helper function to compute a metric. It also checks if the metric function expects the
        arguments `y_true`, `y_pred`, and optionally `sample_weight`.

        Parameters
        ----------
        metric_fn
            Metric function to be used. Note that the loss function must be compatible with the
            `y_true`, `y_pred`, and optionally with `sample_weight`.
        y_true, y_pred, sample_weight
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_metrics`.

        Returns
        -------
        Metric value.
        """
        # get scoring function arguments
        args = inspect.getfullargspec(metric_fn).args

        if 'y_true' not in args:
            raise ValueError('The `scoring` function must have the argument `y_true` in its definition.')

        if 'y_pred' not in args:
            raise ValueError('The `scoring` function must have the argument `y_pred` in its definition.')

        if 'sample_weight' not in args:
            # some metrics might not support `sample_weight` such as:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
            if sample_weight is not None:
                logger.warning(f"The loss function '{metric_fn.__name__}' does not support argument `sample_weight`. "
                               f"Calling the method without `sample_weight`.")

            return metric_fn(y_true=y_true, y_pred=y_pred)  # type: ignore[call-arg]

        # call metric function with all parameters.
        return metric_fn(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)  # type: ignore[call-arg]

    def _compute_permutation_importance(self,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        loss_fns: Dict[
                                            str,
                                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                                        ],
                                        score_fns: Dict[
                                            str,
                                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                                        ],
                                        method: Literal["estimate", "exact"],
                                        kind: Literal["difference", "ratio"],
                                        n_repeats: int,
                                        sample_weight: Optional[np.ndarray],
                                        features: Union[int, Tuple[int, ...]],
                                        loss_orig: Dict[str, float],
                                        score_orig: Dict[str, float]):

        """
        Helper function to compute the permutation importance for a given feature.

        Parameters
        ----------
        X, y, loss_fns, score_fns, method, kind, n_repeats, sample_weight
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.#
        features
            The feature to compute the importance for.
        loss_orig
            Original loss value when the features are left intact. The loss is computed on the original datasets.
        score_orig
            Original score value when the feature are left intact. The score is computed on the original dataset.

        Returns
        --------
        A dictionary having as keys the metric name and as key the feature importance associated with the metric.
        """
        if method == Method.EXACT:
            # computation of the exact statistic which is quadratic in the number of samples
            return self._compute_exact(X=X,
                                       y=y,
                                       loss_fns=loss_fns,
                                       score_fns=score_fns,
                                       kind=kind,
                                       sample_weight=sample_weight,
                                       features=features,
                                       loss_orig=loss_orig,
                                       score_orig=score_orig)

        # sample approximation
        return self._compute_estimate(X=X,
                                      y=y,
                                      loss_fns=loss_fns,
                                      score_fns=score_fns,
                                      kind=kind,
                                      n_repeats=n_repeats,
                                      sample_weight=sample_weight,
                                      features=features,
                                      loss_orig=loss_orig,
                                      score_orig=score_orig)

    def _compute_exact(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       loss_fns: Dict[
                            str,
                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                       ],
                       score_fns: Dict[
                           str,
                           Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                       ],
                       kind: str,
                       sample_weight: Optional[np.ndarray],
                       features: Union[int, Tuple[int, ...]],
                       loss_orig: Dict[str, float],
                       score_orig: Dict[str, float]):
        """
        Helper function to compute the `switch` estimate of the permutation feature importance.

        Parameters
        ----------
        X, y, loss_fns, score_fns, kind, sample_weight, features, loss_orig, score_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the loss name and as key the feature importance associated with the loss.
        """
        y_pred = []
        weights: Optional[List[np.ndarray]] = [] if sample_weight else None

        for i in range(len(X)):
            # create dataset
            X_tmp = np.tile(X[i:i+1], reps=(len(X) - 1, 1))
            X_tmp[:, features] = np.delete(arr=X[:, features], obj=i, axis=0)

            # compute predictions
            y_pred.append(self.predictor(X_tmp))

            # create sample weights if necessary
            if sample_weight is not None:
                weights.append(np.full(shape=(len(X_tmp),), fill_value=sample_weight[i]))  # type: ignore[union-attr]

        # concatenate all predictions and construct ground-truth array. At this point, the `y_pre` vector
        # should contain `N x (N - 1)` predictions, where `N` is the number of samples in `X`.
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.tile(y.reshape(-1, 1), reps=(1, len(X) - 1)).reshape(-1)

        if weights is not None:
            weights = np.concatenate(weights, axis=0)

        # compute loss values for the altered dataset
        loss_permuted = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                               y_true=y_true,
                                                               y_pred=y_pred,  # type: ignore[arg-type]
                                                               sample_weight=weights)  # type: ignore[arg-type]

        # compute score values for the altered dataset
        score_permuted = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                                y_true=y_true,
                                                                y_pred=y_pred,  # type: ignore[arg-type]
                                                                sample_weight=weights)  # type: ignore[arg-type]

        # compute feature importance for the loss functions
        loss_feature_importance = PermutationImportance._compute_importances(metrics_fns=loss_fns,
                                                                             metrics_orig=loss_orig,
                                                                             metrics_permuted=loss_permuted,
                                                                             kind=kind,
                                                                             lower_is_better=True)

        # compute feature importance for the score functions
        score_feature_importance = PermutationImportance._compute_importances(metrics_fns=score_fns,
                                                                              metrics_orig=score_orig,
                                                                              metrics_permuted=score_permuted,
                                                                              kind=kind,
                                                                              lower_is_better=False)

        return {**loss_feature_importance, **score_feature_importance}

    def _compute_estimate(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          loss_fns: Dict[
                              str,
                              Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                          ],
                          score_fns: Dict[
                              str,
                              Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                          ],
                          kind: str,
                          n_repeats: int,
                          sample_weight: Optional[np.ndarray],
                          features: Union[int, Tuple[int, ...]],
                          loss_orig: Dict[str, float],
                          score_orig: Dict[str, float]):
        """
        Helper function to compute the `divide` estimate of the permutation feature importance.

        Parameters
        ----------
        X, y, loss_fns, score_fns, kind, sample_weight, features, loss_orig, score_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the loss name and as key the feature importance associated with the loss.
        """
        N = len(X)
        start, middle, end = 0, N // 2, N if N % 2 == 0 else N - 1
        fh, sh = np.s_[start:middle], np.s_[middle:end]

        loss_permuted: Dict[str, List[float]] = defaultdict(list)
        score_permuted: Dict[str, List[float]] = defaultdict(list)

        for i in range(n_repeats):
            # get random permutation. Note that this includes also the last element
            shuffled_indices = np.random.permutation(len(X))

            # shuffle the dataset
            X_tmp, y_tmp = X[shuffled_indices].copy(), y[shuffled_indices].copy()
            sample_weight_tmp = None if (sample_weight is None) else sample_weight[shuffled_indices].copy()

            # permute values from the first half into the second half and the other way around
            fvals_tmp = X_tmp[fh, features].copy()
            X_tmp[fh, features] = X_tmp[sh, features]
            X_tmp[sh, features] = fvals_tmp

            # compute scores
            y_pred = self.predictor(X_tmp[:end])
            y_true = y_tmp[:end]
            weights = None if (sample_weight_tmp is None) else sample_weight_tmp[:end]

            # compute loss values for the altered dataset
            loss_permuted = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                                   y_true=y_true,
                                                                   y_pred=y_pred,
                                                                   sample_weight=weights,
                                                                   metrics=loss_permuted)

            # compute score values for the altered dataset
            score_permuted = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                                    y_true=y_true,
                                                                    y_pred=y_pred,
                                                                    sample_weight=weights,
                                                                    metrics=score_permuted)

        # compute feature importance for the loss functions
        loss_feature_importance = PermutationImportance._compute_importances(metrics_fns=loss_fns,
                                                                             metrics_orig=loss_orig,
                                                                             metrics_permuted=loss_permuted,
                                                                             kind=kind,
                                                                             lower_is_better=True)

        # compute feature importance for the score functions
        score_feature_importance = PermutationImportance._compute_importances(metrics_fns=score_fns,
                                                                              metrics_orig=score_orig,
                                                                              metrics_permuted=score_permuted,
                                                                              kind=kind,
                                                                              lower_is_better=False)

        return {**loss_feature_importance, **score_feature_importance}

    @staticmethod
    def _compute_importances(metrics_fns: Dict[
                                str,
                                Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                             ],
                             metrics_orig: Dict[str, float],
                             metrics_permuted: Dict[str, List[float]],
                             kind: str,
                             lower_is_better: bool) -> Dict[str, Any]:
        """
        Helper function to compute the feature importance as the metric ration or the metric difference
        based on the `kind` parameter and the `lower_is_better` flag for multiple metric functions.

        Parameters
        ----------
        metrics_fns
            A dictionary of metric functions having as keys the name of the metric function and as
            values the metric function.
        metrics_orig
            A dictionary having as keys the name of the metric and as values a metric values when the
            feature values are left intact.
        metrics_permuted
            A dictionary having as keys the name of the metric and as values a list of metric values when
            the feature values are permuted.
        kind
            Metric value when the feature values are left intact.
        lower_is_better
            Whether lower metric value is better.

        Returns
        -------
        A dictionary having as keys the name of the metric and as values the feature importance or a dictionary
        containing the mean and the standard deviation of the feature importance and the samples used to
        compute the two statistics.
        """
        feature_importance = {}

        for metric_name in metrics_fns:
            importance_values = [
                PermutationImportance._compute_importance(
                    metric_orig=metrics_orig[metric_name],
                    metric_permuted=metric_permuted_value,
                    kind=kind,
                    lower_is_better=lower_is_better
                ) for metric_permuted_value in metrics_permuted[metric_name]
            ]

            if len(importance_values) > 1:
                feature_importance[metric_name] = {
                    "mean": np.mean(importance_values),
                    "std": np.std(importance_values),
                    "samples": np.array(importance_values),
                }
            else:
                feature_importance[metric_name] = importance_values[0].item()

        return feature_importance

    @staticmethod
    def _compute_importance(metric_orig: float, metric_permuted: float, kind: str, lower_is_better: bool):
        """
        Helper function to compute the feature importance as the metric ratio or the metric difference
        based on the `kind` parameter and `lower_is_better` flag.

        Parameters
        ----------
        metric_orig
            Metric value when the feature values are left intact.
        metric_permuted
            Metric value when the feature value are permuted.
        kind
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        lower_is_better
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_importances.

        Returns
        -------
        Importance score.
        """
        if lower_is_better:
            return metric_permuted / metric_orig if kind == Kind.RATIO else metric_permuted - metric_orig

        return metric_orig / metric_permuted if kind == Kind.RATIO else metric_orig - metric_permuted

    def _build_explanation(self,
                           feature_names: List[Union[str, Tuple[str, ...]]],
                           individual_feature_importance: List[
                               Union[
                                   Dict[str, float],
                                   Dict[str, Dict[str, float]]
                               ]
                           ]) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        feature_names
            List of names of the explained features.
        individual_feature_importance
            List of dictionary having as keys the name of the loss function and as values the feature
            importance when ``kind='exact'`` or a dictionary containing the mean and the standard deviation
            when``kind='estimate'``.

        Returns
        -------
        `Explanation` object.

        """
        # list of loss names
        loss_names = list(individual_feature_importance[0].keys())

        # list of lists of features importance, one list per loss function
        feature_importance: List[List[Union[float, Dict[str, float]]]] = []

        for loss_name in loss_names:
            feature_importance.append([])

            for i in range(len(feature_names)):
                feature_importance[-1].append(individual_feature_importance[i][loss_name])

        data = copy.deepcopy(DEFAULT_DATA_PERMUTATION_IMPORTANCE)
        data.update(
            feature_names=feature_names,
            loss_names=loss_names,
            feature_importance=feature_importance,
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


# No type check due to the generic explanation object
@no_type_check
def plot_permutation_importance(exp: Explanation,
                                features: Union[List[int], Literal['all']] = 'all',
                                loss_names: Union[List[Union[str, int]], Literal['all']] = 'all',
                                n_cols: int = 3,
                                sort: bool = True,
                                top_k: int = 10,
                                ax: Optional[Union['plt.Axes', np.ndarray]] = None,
                                bar_kw: Optional[dict] = None,
                                fig_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plot permutation feature importance on `matplotlib` axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method.
    features
        A list of features entries provided in `feature_names` argument  to the
        :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method, or
        ``'all'`` to  plot all the explained features. For example, if  ``feature_names = ['temp', 'hum', 'windspeed']``
        and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    loss_names

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

        .. _matplotlib.pyplot.barh:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    --------
    `plt.Axes` with the feature importance plot.
    """
    from matplotlib.gridspec import GridSpec

    # define figure arguments
    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}

    fig_kw = {**default_fig_kw, **fig_kw}

    # initialize `features` and `loss_names` if set to ``'all'``
    n_features = len(exp.data['feature_names'])
    n_loss_names = len(exp.data['loss_names'])

    if features == 'all':
        features = list(range(n_features))

    if loss_names == 'all':
        loss_names = exp.data['loss_names']

    # `features` sanity checks
    for ifeature in features:
        if ifeature >= len(exp.data['feature_names']):
            raise ValueError(f"The `features` indices must be les thant the "
                             f"``len(feature_names) = {n_features}``. Received {ifeature}.")

    # construct vector of feature names to display importance for
    feature_names = [exp.data['feature_names'][i] for i in features]

    # `loss_names` sanity checks
    for i, iloss_name in enumerate(loss_names):
        if isinstance(iloss_name, str) and (iloss_name not in exp.data['loss_names']):
            raise ValueError(f"Unknown `loss_name`. Received {iloss_name}. "
                             f"Available values are: {exp.data['loss_names']}.")

        if isinstance(iloss_name, numbers.Integral):
            if iloss_name >= n_loss_names:
                raise IndexError(f"Loss name index out of range. Received {iloss_name}. "
                                 f"The number of `loss_names` is {n_loss_names}")

            # convert index to string
            loss_names[i] = exp.data['loss_names'][i]

    if ax is None:
        fix, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_loss_names != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_loss_names)
        n_rows = math.ceil(n_loss_names / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)

        for i, spec in zip(range(n_loss_names), gs):
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)

        if ax.size < n_loss_names:
            raise ValueError(f"Expected ax to have {n_loss_names} axes, got {ax.size}")

        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    for i in range(len(axes_ravel)):
        ax = axes_ravel[i]
        loss_name = loss_names[i]

        # define bar plot data
        y_labels = feature_names
        y_labels = ['(' + ', '.join(y_label) + ')' if isinstance(y_label, tuple) else y_label for y_label in y_labels]

        if exp.meta['params']['method'] == Method.EXACT:
            width = [exp.data['feature_importance'][i][j] for j in features]
            xerr = None
        else:
            width = [exp.data['feature_importance'][i][j]['mean'] for j in features]
            xerr = [exp.data['feature_importance'][i][j]['std'] for j in features]

        if sort:
            sorted_indices = np.argsort(width)[::-1][:top_k]
            width = [width[j] for j in sorted_indices]
            y_labels = [y_labels[j] for j in sorted_indices]

            if exp.meta['params']['method'] == Method.ESTIMATE:
                xerr = [xerr[j] for j in sorted_indices]

        y = np.arange(len(width))
        default_bar_kw = {'align': 'center'}
        bar_kw = default_bar_kw if bar_kw is None else {**default_bar_kw, **bar_kw}

        ax.barh(y=y, width=width, xerr=xerr, **bar_kw)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Permutation feature importance')
        ax.set_title(loss_name)

    fig.set(**fig_kw)
    return axes