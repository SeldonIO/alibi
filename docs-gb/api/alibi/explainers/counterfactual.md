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

### Constructor

```python
Counterfactual(self, predict_fn: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], shape: Tuple[int, ...], distance_fn: str = 'l1', target_proba: float = 1.0, target_class: Union[str, int] = 'other', max_iter: int = 1000, early_stop: int = 50, lam_init: float = 0.1, max_lam_steps: int = 10, tol: float = 0.05, learning_rate_init=0.1, feature_range: Union[Tuple, str] = (-10000000000.0, 10000000000.0), eps: Union[float, numpy.ndarray] = 0.01, init: str = 'identity', decay: bool = True, write_dir: Optional[str] = None, debug: bool = False, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict_fn` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  | `tensorflow` model or any other model's prediction function returning class probabilities. |
| `shape` | `Tuple[int, .Ellipsis]` |  | Shape of input data starting with batch size. |
| `distance_fn` | `str` | `'l1'` | Distance function to use in the loss term. |
| `target_proba` | `float` | `1.0` | Target probability for the counterfactual to reach. |
| `target_class` | `Union[str, int]` | `'other'` | Target class for the counterfactual to reach, one of ``'other'``, ``'same'`` or an integer denoting desired class membership for the counterfactual instance. |
| `max_iter` | `int` | `1000` | Maximum number of iterations to run the gradient descent for (inner loop). |
| `early_stop` | `int` | `50` | Number of steps after which to terminate gradient descent if all or none of found instances are solutions. |
| `lam_init` | `float` | `0.1` | Initial regularization constant for the prediction part of the Wachter loss. |
| `max_lam_steps` | `int` | `10` | Maximum number of times to adjust the regularization constant (outer loop) before terminating the search. |
| `tol` | `float` | `0.05` | Tolerance for the counterfactual target probability. |
| `learning_rate_init` |  | `0.1` | Initial learning rate for each outer loop of `lambda`. |
| `feature_range` | `Union[Tuple, str]` | `(-10000000000.0, 10000000000.0)` | Tuple with `min` and `max` ranges to allow for perturbed instances. `Min` and `max` ranges can be `float` or `numpy` arrays with dimension (1 x nb of features) for feature-wise ranges. |
| `eps` | `Union[float, numpy.ndarray]` | `0.01` | Gradient step sizes used in calculating numerical gradients, defaults to a single value for all features, but can be passed an array for feature-wise step sizes. |
| `init` | `str` | `'identity'` | Initialization method for the search of counterfactuals, currently must be ``'identity'``. |
| `decay` | `bool` | `True` | Flag to decay learning rate to zero for each outer loop over lambda. |
| `write_dir` | `Optional[str]` | `None` | Directory to write `tensorboard` files to. |
| `debug` | `bool` | `False` | Flag to write `tensorboard` summaries for debugging. |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` | Optional `tensorflow` session that will be used if passed instead of creating or inferring one internally. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to be explained. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(X: numpy.ndarray, y: Optional[numpy.ndarray]) -> alibi.explainers.counterfactual.Counterfactual
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Not used. Included for consistency. |
| `y` | `Optional[numpy.ndarray]` |  | Not used. Included for consistency. |

**Returns**
- Type: `alibi.explainers.counterfactual.Counterfactual`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  | New predictor function/model. |

**Returns**
- Type: `None`

## Functions
### `CounterFactual`

```python
CounterFactual(args, kwargs)
```

The class name `CounterFactual` is deprecated, please use `Counterfactual`.
