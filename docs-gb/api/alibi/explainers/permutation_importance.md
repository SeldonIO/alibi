# `alibi.explainers.permutation_importance`
## `Kind`

_Inherits from:_ `str`, `Enum`

Enumeration of supported kind.

## `Method`

_Inherits from:_ `str`, `Enum`

Enumeration of supported method.

## `PermutationImportance`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Implementation of the permutation feature importance for tabular datasets. The method measure the importance

of a feature as the relative increase/decrease in the loss/score function when the feature values are permuted.
Supports black-box models.

For details of the method see the papers:

 - https://link.springer.com/article/10.1023/A:1010933404324

 - https://arxiv.org/abs/1801.01489

### Constructor

```python
PermutationImportance(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], loss_fns: Union[Literal['mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'log_loss'], List[Literal['mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'log_loss']], Callable[[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]], float], Dict[str, Callable[[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]], float]], NoneType] = None, score_fns: Union[Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2'], List[Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2']], Callable[[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]], float], Dict[str, Callable[[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]], float]], NoneType] = None, feature_names: Optional[List[str]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `loss_fns` | `Union[Literal[mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, log_loss], List[Literal[mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, log_loss]], Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float], Dict[str, Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float]], None]` | `None` |  |
| `score_fns` | `Union[Literal[accuracy, precision, recall, f1, roc_auc, r2], List[Literal[accuracy, precision, recall, f1, roc_auc, r2]], Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float], Dict[str, Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float]], None]` | `None` |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, y: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, .Ellipsis]]]] = None, method: Literal[estimate, exact] = 'estimate', kind: Literal[ratio, difference] = 'ratio', n_repeats: int = 50, sample_weight: Optional[numpy.ndarray] = None) -> alibi.api.interfaces.Explanation
```

Computes the permutation feature importance for each feature with respect to the given loss or score

functions and the dataset `(X, y)`.

Parameters
----------
X
    A `N x F` input feature dataset used to calculate the permutation feature importance. This is typically the
    test dataset.
y
    Ground-truth labels array  of size `N` (i.e. `(N, )`) corresponding the input feature `X`.
features
    An optional list of features or tuples of features for which to compute the permutation feature
    importance. If not provided, the permutation feature importance will be computed for every single features
    in the dataset. Some example of `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``,
    where ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
method
    The method to be used to compute the feature importance. If set to ``'exact'``, a "switch" operation is
    performed across all observed pairs, by excluding pairings that are actually observed in the original
    dataset. This operation is quadratic in the number of samples (`N x (N - 1)` samples) and thus can be
    computationally intensive. If set to ``'estimate'``, the dataset will be divided in half. The values of
    the first half containing the ground-truth labels the rest of the features (i.e. features that are left
    intact) is matched with the values of the second half of the permuted features, and the other way around.
    This method is computationally lighter and provides estimate error bars given by the standard deviation.
    Note that for some specific loss and score functions, the estimate does not converge to the exact metric
    value.
kind
    Whether to report the importance as the loss/score ratio or the loss/score difference.
    Available values are: ``'ratio'`` | ``'difference'``.
n_repeats
    Number of times to permute the feature values. Considered only when ``method='estimate'``.
sample_weight
    Optional weight for each sample instance.

Returns
-------
explanation
    An `Explanation` object containing the data and the metadata of the permutation feature importance.
    See usage at `Permutation feature importance examples`_ for details

    .. _Permutation feature importance examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `y` | `numpy.ndarray` |  |  |
| `features` | `Optional[List[Union[int, Tuple[int, .Ellipsis]]]]` | `None` |  |
| `method` | `Literal[estimate, exact]` | `'estimate'` |  |
| `kind` | `Literal[ratio, difference]` | `'ratio'` |  |
| `n_repeats` | `int` | `50` |  |
| `sample_weight` | `Optional[numpy.ndarray]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

Parameters
----------
predictor
    New predictor function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

## Functions
### `plot_permutation_importance`

```python
plot_permutation_importance(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', metric_names: Union[List[Union[str, int]], Literal[all]] = 'all', n_cols: int = 3, sort: bool = True, top_k: Optional[int] = None, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, bar_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> plt.Axes
```

Plot permutation feature importance on `matplotlib` axes.

Parameters
----------
exp
    An `Explanation` object produced by a call to the
    :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method.
features
    A list of feature entries provided in `feature_names` argument  to the
    :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method, or
    ``'all'`` to  plot all the explained features. For example, consider that the
    ``feature_names = ['temp', 'hum', 'windspeed', 'season']``. If we set `features=None` in the `explain` method,
    meaning that all the feature were explained, and we want to plot only the values  for the ``'temp'`` and
    ``'windspeed'``, then we would set ``features=[0, 2]``. Otherwise, if we set `features=[1, 2, 3]` in the
    explain method, meaning that we explained ``['hum', 'windspeed', 'season']``, and we want to plot the values
    only for ``['windspeed', 'season']``, then we would set ``features=[1, 2]`` (i.e., their index in the
    `features` list passed to the `explain` method). Defaults to ``'all'``.
metric_names
    A list of metric entries in the `exp.data['metrics']` to plot the permutation feature importance for,
    or ``'all'`` to plot the permutation feature importance for all metrics (i.e., loss and score functions).
    The ordering is given by the concatenation of the loss metrics followed by the score metrics.
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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  |  |
| `features` | `Union[List[int], Literal[all]]` | `'all'` |  |
| `metric_names` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` |  |
| `n_cols` | `int` | `3` |  |
| `sort` | `bool` | `True` |  |
| `top_k` | `Optional[int]` | `None` |  |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` |  |
| `bar_kw` | `Optional[dict]` | `None` |  |
| `fig_kw` | `Optional[dict]` | `None` |  |

**Returns**
- Type: `plt.Axes`
