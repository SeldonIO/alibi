# `alibi.explainers.permutation_importance`
## Constants
### `DEFAULT_DATA_PERMUTATION_IMPORTANCE`
```python
DEFAULT_DATA_PERMUTATION_IMPORTANCE: dict = {'feature_names': None, 'metric_names': None, 'feature_importance': None}
```

### `DEFAULT_META_PERMUTATION_IMPORTANCE`
```python
DEFAULT_META_PERMUTATION_IMPORTANCE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.permutation_importance (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

### `LOSS_FNS`
```python
LOSS_FNS: dict = {'mean_absolute_error': <function mean_absolute_error at 0x1603b4c10>, 'mean_...
```

### `SCORE_FNS`
```python
SCORE_FNS: dict = {'accuracy': <function accuracy_score at 0x160380700>, 'precision': <function...
```

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
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A prediction function which receives as input a `numpy` array of size `N x F`, and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances, `F` is the number of features, and `T` is the number of targets. Note that the output shape must be compatible with the loss and score functions provided in `loss_fns` and `score_fns`. |
| `loss_fns` | `Union[Literal[mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, log_loss], List[Literal[mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, log_loss]], Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float], Dict[str, Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float]], None]` | `None` | A literal, or a list of literals, or a loss function, or a dictionary of loss functions having as keys the names of the loss functions and as values the loss functions (i.e., lower values are better). The available literal values are described in :py:data:`alibi.explainers.permutation_importance.LOSS_FNS`. Note that the `predictor` output must be compatible with every loss function. Every loss function is expected to receive the following arguments: - `y_true` : ``np.ndarray`` -  a `numpy` array of ground-truth labels. - `y_pred` | `y_score` : ``np.ndarray`` - a `numpy` array of model predictions. This corresponds to              the output of the model. - `sample_weight`: ``Optional[np.ndarray]`` - a `numpy` array of sample weights. |
| `score_fns` | `Union[Literal[accuracy, precision, recall, f1, roc_auc, r2], List[Literal[accuracy, precision, recall, f1, roc_auc, r2]], Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float], Dict[str, Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, typing.Optional[numpy.ndarray]]], float]], None]` | `None` | A literal, or a list or literals, or a score function, or a dictionary of score functions having as keys the names of the score functions and as values the score functions (i.e, higher values are better). The available literal values are described in :py:data:`alibi.explainers.permutation_importance.SCORE_FNS`. As with the `loss_fns`, the `predictor` output must be compatible with every score function and the score function must have the same signature presented in the `loss_fns` parameter description. |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, y: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, .Ellipsis]]]] = None, method: Literal[estimate, exact] = 'estimate', kind: Literal[ratio, difference] = 'ratio', n_repeats: int = 50, sample_weight: Optional[numpy.ndarray] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | A `N x F` input feature dataset used to calculate the permutation feature importance. This is typically the test dataset. |
| `y` | `numpy.ndarray` |  | Ground-truth labels array  of size `N` (i.e. `(N, )`) corresponding the input feature `X`. |
| `features` | `Optional[List[Union[int, Tuple[int, .Ellipsis]]]]` | `None` | An optional list of features or tuples of features for which to compute the permutation feature importance. If not provided, the permutation feature importance will be computed for every single features in the dataset. Some example of `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively. |
| `method` | `Literal[estimate, exact]` | `'estimate'` | The method to be used to compute the feature importance. If set to ``'exact'``, a "switch" operation is performed across all observed pairs, by excluding pairings that are actually observed in the original dataset. This operation is quadratic in the number of samples (`N x (N - 1)` samples) and thus can be computationally intensive. If set to ``'estimate'``, the dataset will be divided in half. The values of the first half containing the ground-truth labels the rest of the features (i.e. features that are left intact) is matched with the values of the second half of the permuted features, and the other way around. This method is computationally lighter and provides estimate error bars given by the standard deviation. Note that for some specific loss and score functions, the estimate does not converge to the exact metric value. |
| `kind` | `Literal[ratio, difference]` | `'ratio'` | Whether to report the importance as the loss/score ratio or the loss/score difference. Available values are: ``'ratio'`` | ``'difference'``. |
| `n_repeats` | `int` | `50` | Number of times to permute the feature values. Considered only when ``method='estimate'``. |
| `sample_weight` | `Optional[numpy.ndarray]` | `None` | Optional weight for each sample instance. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New predictor function. |

**Returns**
- Type: `None`

## Functions
### `plot_permutation_importance`

```python
plot_permutation_importance(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', metric_names: Union[List[Union[str, int]], Literal[all]] = 'all', n_cols: int = 3, sort: bool = True, top_k: Optional[int] = None, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, bar_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> plt.Axes
```

Plot permutation feature importance on `matplotlib` axes.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method. |
| `features` | `Union[List[int], Literal[all]]` | `'all'` | A list of feature entries provided in `feature_names` argument  to the :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method, or ``'all'`` to  plot all the explained features. For example, consider that the ``feature_names = ['temp', 'hum', 'windspeed', 'season']``. If we set `features=None` in the `explain` method, meaning that all the feature were explained, and we want to plot only the values  for the ``'temp'`` and ``'windspeed'``, then we would set ``features=[0, 2]``. Otherwise, if we set `features=[1, 2, 3]` in the explain method, meaning that we explained ``['hum', 'windspeed', 'season']``, and we want to plot the values only for ``['windspeed', 'season']``, then we would set ``features=[1, 2]`` (i.e., their index in the `features` list passed to the `explain` method). Defaults to ``'all'``. |
| `metric_names` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` | A list of metric entries in the `exp.data['metrics']` to plot the permutation feature importance for, or ``'all'`` to plot the permutation feature importance for all metrics (i.e., loss and score functions). The ordering is given by the concatenation of the loss metrics followed by the score metrics. |
| `n_cols` | `int` | `3` | Number of columns to organize the resulting plot into. |
| `sort` | `bool` | `True` | Boolean flag whether to sort the values in descending order. |
| `top_k` | `Optional[int]` | `None` | Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed. |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` | A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on. |
| `bar_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.barh`_ function. |
| `fig_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.figure.set`_ function. .. _matplotlib.pyplot.barh: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html .. _matplotlib.figure.set: https://matplotlib.org/stable/api/figure_api.html |

**Returns**
- Type: `plt.Axes`
