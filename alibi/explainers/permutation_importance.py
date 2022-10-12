import numbers
import sys
import copy
import logging

import inspect
import numpy as np
import math
from enum import Enum
from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_PERMUTATION_IMPORTANCE, DEFAULT_DATA_PERMUTATION_IMPORTANCE
from typing import Callable, Optional, Union, List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    https://link.springer.com/article/10.1023/A:1010933404324
    https://arxiv.org/abs/1801.01489
    """

    def __init__(self,
                 predictor: Callable[[np.array], np.array],
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

    def explain(self,
                X: np.ndarray,
                y: np.ndarray,
                loss_fns: Union
                    [
                      Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                      Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ],
                features: Optional[List[int]] = None,
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
            A list of feature for which to compute the feature importance for. An example would be: ``[0, 1, 3]``,
            where ``0``, ``1``, and ``3`` correspond to the columns 0, 1, and 3 in `X`. If not provide, the feature
            importance will be computed for every single feature in the dataset.
        method
            The method to be used to compute the feature importance. If set to``'switch'``, a "switch" operation is
            performed across all observed pairs, by excluding pairings that are actually observed in the original
            dataset. This operation is quadratic in the number of samples (`N x (N - 1)` samples) and thus can be
            computationally intensive. If set to ``'divide'``, the dataset will be divided in half and the first
            half's values of the remaining ground-truth labels and the rest of the feature is matched with the
            second half's values of the permuted features, and the other way around. This method is computationally
            lighter and provides estimate error bars given by the standard deviation.
        kind
            Whether to report the importance as the error ratio or the error difference. Available values are:
            ``'ratio'` | ``'difference'``.
        n_repeats
            Number of times to permute the features. Considered only when ``method='divide'``.
        sample_weight
            Optional weight for each sample instance.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated permutation feature
            importance. See usage at `Permutation importance examples`_ for details

            .. _Permutation feature importance examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html
        """
        n_features = X.shape[1]

        if callable(loss_fns):
            loss_fns = {loss_fns.__name__: loss_fns}

        # set the `features_names` when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # construct `feature_names` based on the `features`. If `features` is ``None``, then initialize
        # `features` with all single feature available in the dataset.
        if features:
            feature_names = [self.feature_names[f] for f in features]
        else:
            feature_names = self.feature_names
            features = list(range(n_features))

        # compute the base score
        loss_orig = {}

        for loss_name, loss_fn in loss_fns.items():
            loss_orig[loss_name] = self._compute_loss(y_true=y,
                                                      y_pred=self.predictor(X),
                                                      loss_fn=loss_fn,
                                                      sample_weight=sample_weight)

        # compute permutation feature importance for every feature
        # TODO: implement parallel version - future work as it can be done for ALE too
        individual_feature_importance = []

        for ifeature in tqdm(features, disable=not self.verbose):
            individual_feature_importance.append(
                self._compute_permutation_importance(
                    X=X,
                    y=y,
                    loss_fns=loss_fns,
                    method=method,
                    kind=kind,
                    n_repeats=n_repeats,
                    sample_weight=sample_weight,
                    feature=ifeature,
                    loss_orig=loss_orig,
                )
            )

        # update meta data params
        self.meta['params'].update(feature_names=feature_names,
                                   method=method,
                                   kind=kind,
                                   n_repeats=n_repeats,
                                   sample_weight=sample_weight)

        # build and return the explanation object
        return self._build_explanation(feature_names=feature_names,
                                       individual_feature_importance=individual_feature_importance)

    def _compute_loss(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      loss_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                      sample_weight: Optional[np.ndarray]) -> float:
        """
        Helper function to compute the loss. It also checks if the loss function expects the arguments `y_true`,
        `y_pred`, and optionally `sample_weight`.

        Parameters
        ----------
        y_true
            Ground truth targets.
        y_pred
            Predicted outcome as returned by the classifier.
        loss_fn
            Loss function to be used. Note that the loss function must be compatible with the `y_true`, `y_pred`, and
            optionally with `sample_weight`.
        sample_weight
            Weight of each sample instance.

        Returns
        -------
        Loss value.
        """
        # get scoring function arguments
        args = inspect.getfullargspec(loss_fn).args

        if 'y_true' not in args:
            raise ValueError('The `scoring` function must have the argument `y_true` in its definition.')

        if 'y_pred' not in args:
            raise ValueError('The `scoring` function must have the argument `y_pred` in its definition.')

        if 'sample_weight' not in args:
            # some scores might not support `sample_weight` such as:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
            if sample_weight is not None:
                logger.warning(f"The loss function '{loss_fn.__name__}' does not support argument `sample_weight`. "
                               f"Calling the method without `sample_weight`.")

            return loss_fn(y_true=y_true, y_pred=y_pred)

        # call scoring function with all parameters.
        return loss_fn(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def _compute_permutation_importance(self,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                                        method: Literal["estimate", "exact"],
                                        kind: Literal["difference", "ratio"],
                                        n_repeats: int,
                                        sample_weight: Optional[np.ndarray],
                                        feature: int,
                                        loss_orig: Dict[str, float]):

        """
        Helper function to compute the permutation importance for a given feature.

        Parameters
        ----------
        X, y, loss_fns, method, kind, n_repeats, sample_weight
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.#
        feature
            The feature to compute the importance for.
        loss_orig
            Original loss value when the features are left intact. The loss is computed on the original datasets.

        Returns
        --------
        A dictionary having as keys the loss name and as key the feature importance associated with the loss.
        """
        if method == Method.EXACT:
            # computation of the exact statistic which is quadratic in the number of samples
            return self._compute_exact(X=X,
                                       y=y,
                                       loss_fns=loss_fns,
                                       kind=kind,
                                       sample_weight=sample_weight,
                                       feature=feature,
                                       loss_orig=loss_orig)
        # sample approximation
        return self._compute_estimate(X=X,
                                      y=y,
                                      loss_fns=loss_fns,
                                      kind=kind,
                                      n_repeats=n_repeats,
                                      sample_weight=sample_weight,
                                      feature=feature,
                                      loss_orig=loss_orig)

    def _compute_exact(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                       kind: str,
                       sample_weight: Optional[np.ndarray],
                       feature: int,
                       loss_orig: Dict[str, float]):
        """
        Helper function to compute the `switch` estimate of the permutation feature importance.

        Parameters
        ----------
        X, y, loss_fns, kind, sample_weight, feature, loss_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the loss name and as key the feature importance associated with the loss.
        """

        y_pred = []
        weights = [] if sample_weight else None
        loss_permuted = {}

        for i in range(len(X)):
            # create dataset
            X_tmp = np.tile(X[i:i+1], reps=(len(X) - 1, 1))
            X_tmp[:, feature] = np.delete(arr=X[:, feature], obj=i, axis=0)

            # compute predictions
            y_pred.append(self.predictor(X_tmp))

            # create sample weights if necessary
            if sample_weight is not None:
                weights.append(np.full(shape=(len(X_tmp),), fill_value=sample_weight[i]))

        # concatenate all predictions and construct ground-truth array. At this point, the `y_pre` vector
        # should contain `N x (N - 1)` predictions, where `N` is the number of samples in `X`.
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.tile(y.reshape(-1, 1), reps=(1, len(X) - 1)).reshape(-1)

        for loss_name, loss_fn in loss_fns.items():
            loss_permuted[loss_name] = self._compute_loss(y_true=y_true,
                                                          y_pred=y_pred,
                                                          sample_weight=weights,
                                                          loss_fn=loss_fn)

        return {loss_name: self._compute_importance(
            loss_orig=loss_orig[loss_name],
            loss_permuted=loss_permuted[loss_name],
            kind=kind) for loss_name in loss_fns
        }

    def _compute_estimate(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          loss_fns: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
                          kind: str,
                          n_repeats: int,
                          sample_weight: Optional[np.ndarray],
                          feature: int,
                          loss_orig: Dict[str, float]):
        """
        Helper function to compute the `divide` estimate of the permutation feature importance.

        Parameters
        ----------
        X, y, loss_fns, kind, sample_weight, feature, loss_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the loss name and as key the feature importance associated with the loss.
        """
        N = len(X)
        start, middle, end = 0, N // 2, N if N % 2 == 0 else N - 1
        fh, sh = np.s_[start:middle], np.s_[middle:end]
        loss_permuted = {loss_name: [] for loss_name in loss_fns}

        for i in range(n_repeats):
            # get random permutation. Note that this includes also the last element
            shuffled_indices = np.random.permutation(len(X))

            # shuffle the dataset
            X_tmp, y_tmp = X[shuffled_indices].copy(), y[shuffled_indices].copy()
            sample_weight_tmp = None if (sample_weight is None) else sample_weight[shuffled_indices].copy()

            # permute values from the first half into the second half and the other way around
            fvals_tmp = X_tmp[fh, feature].copy()
            X_tmp[fh, feature] = X_tmp[sh, feature]
            X_tmp[sh, feature] = fvals_tmp

            # compute scores
            y_pred = self.predictor(X_tmp[:end])
            y_true = y_tmp[:end]
            weights = None if (sample_weight_tmp is None) else sample_weight_tmp[:end]

            for loss_name, loss_fn in loss_fns.items():
                loss_permuted[loss_name].append(
                    self._compute_loss(y_true=y_true,
                                       y_pred=y_pred,
                                       sample_weight=weights,
                                       loss_fn=loss_fn)
                )

        feature_importance = {}

        for loss_name in loss_fns:
            importance_values = [
                self._compute_importance(
                    loss_orig=loss_orig[loss_name],
                    loss_permuted=loss_permuted_value,
                    kind=kind,
                ) for loss_permuted_value in loss_permuted[loss_name]
            ]

            feature_importance[loss_name] = {
                "mean": np.mean(importance_values),
                "std": np.std(importance_values)
            }

        return feature_importance

    @staticmethod
    def _compute_importance(loss_orig: float, loss_permuted: float, kind):
        """
        Helper function to compute the feature importance as the error ratio or the error difference
        based on the `kind` parameter.

        Parameters
        ----------
        loss_orig
            Loss value when the feature values are left intact.
        loss_permuted
            Loss value when the feature value are permuted.
        kind
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.

        Returns
        -------
        Importance score.
        """
        return loss_permuted / loss_orig if kind == Kind.RATIO else loss_permuted - loss_orig

    def _build_explanation(self,
                           feature_names: List[str],
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
            List of dictionary having as keys the name of the loss function and as values the feature importance
            when ``kind='exact'`` or a dictionary containing the mean and the standard deviation when
            ``kind='estimate'``.

        Returns
        -------
        `Explanation` object.

        """
        print(feature_names)

        # list of loss names
        loss_names = list(individual_feature_importance[0].keys())

        # list of lists of features importance, one list per target
        feature_importance = []
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


def plot_permutation_importance(exp: Explanation,
                                features: Union[List[int], Literal['all']] = 'all',
                                loss_names: Union[List[Union[str, int]], Literal['all']] = 'all',
                                n_cols: int = 3,
                                sort: bool = True,
                                top_k: bool = 10,
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

        if exp.meta['params']['method'] == Method.EXACT:
            width = [exp.data['feature_importance'][i][j] for j in features]
            xerr = None
        else:
            width = [exp.data['feature_importance'][i][j]['mean'] for j in features]
            xerr = [exp.data['feature_importance'][i][j]['std'] for j in features]

        if sort:
            sorted_indices = np.argsort(width)[::-1][:top_k]
            width = [width[j] for j in sorted_indices]
            xerr = [xerr[j] for j in sorted_indices]
            y_labels = [y_labels[j] for j in sorted_indices]

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