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
