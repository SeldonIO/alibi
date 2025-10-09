# `alibi.confidence.model_linearity`
## `LinearityMeasure`

### Constructor

```python
LinearityMeasure(self, method: str = 'grid', epsilon: float = 0.04, nb_samples: int = 10, res: int = 100, alphas: Optional[numpy.ndarray] = None, model_type: str = 'classifier', agg: str = 'pairwise', verbose: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `method` | `str` | `'grid'` |  |
| `epsilon` | `float` | `0.04` |  |
| `nb_samples` | `int` | `10` |  |
| `res` | `int` | `100` |  |
| `alphas` | `Optional[numpy.ndarray]` | `None` |  |
| `model_type` | `str` | `'classifier'` |  |
| `agg` | `str` | `'pairwise'` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `fit`

```python
fit(X_train: numpy.ndarray) -> None
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `numpy.ndarray` |  |  |

**Returns**
- Type: `None`

#### `score`

```python
score(predict_fn: Callable, x: numpy.ndarray) -> numpy.ndarray
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict_fn` | `Callable` |  |  |
| `x` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

## Functions
### `infer_feature_range`

```python
infer_feature_range(X_train: numpy.ndarray) -> numpy.ndarray
```

Infers the feature range from the training set.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `linearity_measure`

```python
linearity_measure(predict_fn: Callable, x: numpy.ndarray, feature_range: Union[List[Any], numpy.ndarray, None] = None, method: str = 'grid', X_train: Optional[numpy.ndarray] = None, epsilon: float = 0.04, nb_samples: int = 10, res: int = 100, alphas: Optional[numpy.ndarray] = None, agg: str = 'global', model_type: str = 'classifier') -> numpy.ndarray
```

Calculate the linearity measure of the model around an instance of interest x.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict_fn` | `Callable` |  |  |
| `x` | `numpy.ndarray` |  |  |
| `feature_range` | `Union[List[Any], numpy.ndarray, None]` | `None` |  |
| `method` | `str` | `'grid'` |  |
| `X_train` | `Optional[numpy.ndarray]` | `None` |  |
| `epsilon` | `float` | `0.04` |  |
| `nb_samples` | `int` | `10` |  |
| `res` | `int` | `100` |  |
| `alphas` | `Optional[numpy.ndarray]` | `None` |  |
| `agg` | `str` | `'global'` |  |
| `model_type` | `str` | `'classifier'` |  |

**Returns**
- Type: `numpy.ndarray`
