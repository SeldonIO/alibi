# `alibi.confidence.model_linearity`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.confidence.model_linearity (WARNING)>
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

### `fit`

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

### `score`

```python
score(predict_fn: Callable, x: numpy.ndarray) -> numpy.ndarray
```

Parameters

----------
predict_fn
    Prediction function.
x
    Instance of interest.

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
