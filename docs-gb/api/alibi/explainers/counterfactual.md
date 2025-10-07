# `alibi.explainers.counterfactual`
## Constants
### `DEFAULT_DATA_CF`
```python
DEFAULT_DATA_CF: dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'success': None}
```

### `DEFAULT_META_CF`
```python
DEFAULT_META_CF: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.counterfactual (WARNING)>
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

## `Counterfactual`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Constructor

```python
Counterfactual(self, predict_fn: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], shape: Tuple[int, ...], distance_fn: str = 'l1', target_proba: float = 1.0, target_class: Union[str, int] = 'other', max_iter: int = 1000, early_stop: int = 50, lam_init: float = 0.1, max_lam_steps: int = 10, tol: float = 0.05, learning_rate_init=0.1, feature_range: Union[Tuple, str] = (-10000000000.0, 10000000000.0), eps: Union[float, numpy.ndarray] = 0.01, init: str = 'identity', decay: bool = True, write_dir: Optional[str] = None, debug: bool = False, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict_fn` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  |  |
| `shape` | `Tuple[int, .Ellipsis]` |  |  |
| `distance_fn` | `str` | `'l1'` |  |
| `target_proba` | `float` | `1.0` |  |
| `target_class` | `Union[str, int]` | `'other'` |  |
| `max_iter` | `int` | `1000` |  |
| `early_stop` | `int` | `50` |  |
| `lam_init` | `float` | `0.1` |  |
| `max_lam_steps` | `int` | `10` |  |
| `tol` | `float` | `0.05` |  |
| `learning_rate_init` |  | `0.1` |  |
| `feature_range` | `Union[Tuple, str]` | `(-10000000000.0, 10000000000.0)` |  |
| `eps` | `Union[float, numpy.ndarray]` | `0.01` |  |
| `init` | `str` | `'identity'` |  |
| `decay` | `bool` | `True` |  |
| `write_dir` | `Optional[str]` | `None` |  |
| `debug` | `bool` | `False` |  |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` |  |

### Methods

### `explain`

```python
explain(X: numpy.ndarray) -> alibi.api.interfaces.Explanation
```

Explain an instance and return the counterfactual with metadata.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

### `fit`

```python
fit(X: numpy.ndarray, y: Optional[numpy.ndarray]) -> alibi.explainers.counterfactual.Counterfactual
```

Fit method - currently unused as the counterfactual search is fully unsupervised.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `y` | `Optional[numpy.ndarray]` |  |  |

**Returns**
- Type: `alibi.explainers.counterfactual.Counterfactual`

### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

Resets the predictor function/model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  |  |

**Returns**
- Type: `None`

## Functions
### `CounterFactual`

```python
CounterFactual(args, kwargs)
```

The class name `CounterFactual` is deprecated, please use `Counterfactual`.
