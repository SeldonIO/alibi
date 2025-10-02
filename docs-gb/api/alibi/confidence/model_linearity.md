# `alibi.confidence.model_linearity`
## Classes
### `LinearityMeasure`

#### Constructor

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

#### Methods

##### `fit`

```python
fit(X_train: numpy.ndarray) -> None
```

Parameters

----------
X_train
    Training set.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `numpy.ndarray` |  |  |

**Returns**
- Type: `None`

##### `score`

```python
score(predict_fn: Callable, x: numpy.ndarray) -> numpy.ndarray
```

Parameters

----------
predict_fn
    Prediction function.
x
    Instance of interest.

Returns
-------
Linearity measure.

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
----------
X_train
    Training set.

Returns
-------
Feature range.

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
----------
predict_fn
    Predict function.
x
    Instance of interest.
feature_range
    Array with min and max values for each feature.
method
    Method for sampling. Supported values: ``'knn'`` | ``'grid'``.
X_train
    Training set.
epsilon
    Size of the sampling region as a percentage of the feature range.
nb_samples
    Number of samples to generate.
res
    Resolution of the grid. Number of intervals in which the features range is discretized.
alphas
    Coefficients in the superposition.
agg
    Aggregation method. Supported values: ``'global'`` | ``'pairwise'``.
model_type
    Type of task. Supported values: ``'regressor'`` | ``'classifier'``.

Returns
-------
Linearity measure.

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
