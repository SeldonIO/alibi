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
LOSS_FNS: dict = {'mean_absolute_error': <function mean_absolute_error at 0x13ac4bd30>, 'mean_...
```

### `SCORE_FNS`
```python
SCORE_FNS: dict = {'accuracy': <function accuracy_score at 0x13a8da820>, 'precision': <function...
```

## `Kind`

_Inherits from:_ `str`, `Enum`

Enumeration of supported kind.

### Constructor

```python
Kind(self, /, *args, **kwargs)
```

## `Method`

_Inherits from:_ `str`, `Enum`

Enumeration of supported method.

### Constructor

```python
Method(self, /, *args, **kwargs)
```

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

### `explain`

```python
explain(X: numpy.ndarray, y: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, .Ellipsis]]]] = None, method: Literal[estimate, exact] = 'estimate', kind: Literal[ratio, difference] = 'ratio', n_repeats: int = 50, sample_weight: Optional[numpy.ndarray] = None) -> alibi.api.interfaces.Explanation
```

Computes the permutation feature importance for each feature with respect to the given loss or score

functions and the dataset `(X, y)`.

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

### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

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
