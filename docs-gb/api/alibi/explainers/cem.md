# `alibi.explainers.cem`
## Constants
### `DEFAULT_DATA_CEM`
```python
DEFAULT_DATA_CEM: dict = {'PN': None, 'PP': None, 'PN_pred': None, 'PP_pred': None, 'grads_graph': Non...
```

### `DEFAULT_META_CEM`
```python
DEFAULT_META_CEM: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.cem (WARNING)>
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

## `CEM`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
CEM(self, predict: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], mode: str, shape: tuple, kappa: float = 0.0, beta: float = 0.1, feature_range: tuple = (-10000000000.0, 10000000000.0), gamma: float = 0.0, ae_model: Optional[keras.src.models.model.Model] = None, learning_rate_init: float = 0.01, max_iterations: int = 1000, c_init: float = 10.0, c_steps: int = 10, eps: tuple = (0.001, 0.001), clip: tuple = (-100.0, 100.0), update_num_grad: int = 1, no_info_val: Union[float, numpy.ndarray, NoneType] = None, write_dir: Optional[str] = None, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  | `tensorflow` model or any other model's prediction function returning class probabilities. |
| `mode` | `str` |  | Find pertinent negatives (PN) or pertinent positives (PP). |
| `shape` | `tuple` |  | Shape of input data starting with batch size. |
| `kappa` | `float` | `0.0` | Confidence parameter for the attack loss term. |
| `beta` | `float` | `0.1` | Regularization constant for L1 loss term. |
| `feature_range` | `tuple` | `(-10000000000.0, 10000000000.0)` | Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be `float` or `numpy` arrays with dimension (1x nb of features) for feature-wise ranges. |
| `gamma` | `float` | `0.0` | Regularization constant for optional auto-encoder loss term. |
| `ae_model` | `Optional[keras.src.models.model.Model]` | `None` | Optional auto-encoder model used for loss regularization. |
| `learning_rate_init` | `float` | `0.01` | Initial learning rate of optimizer. |
| `max_iterations` | `int` | `1000` | Maximum number of iterations for finding a PN or PP. |
| `c_init` | `float` | `10.0` | Initial value to scale the attack loss term. |
| `c_steps` | `int` | `10` | Number of iterations to adjust the constant scaling the attack loss term. |
| `eps` | `tuple` | `(0.001, 0.001)` | If numerical gradients are used to compute `dL/dx = (dL/dp) * (dp/dx)`, then `eps[0]` is used to calculate `dL/dp` and `eps[1]` is used for `dp/dx`. `eps[0]` and `eps[1]` can be a combination of `float` values and `numpy` arrays. For `eps[0]`, the array dimension should be (1x nb of prediction categories) and for `eps[1]` it should be (1x nb of features). |
| `clip` | `tuple` | `(-100.0, 100.0)` | Tuple with `min` and `max` clip ranges for both the numerical gradients and the gradients obtained from the `tensorflow` graph. |
| `update_num_grad` | `int` | `1` | If numerical gradients are used, they will be updated every `update_num_grad` iterations. |
| `no_info_val` | `Union[float, numpy.ndarray, None]` | `None` | Global or feature-wise value considered as containing no information. |
| `write_dir` | `Optional[str]` | `None` | Directory to write `tensorboard` files to. |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` | Optional `tensorflow` session that will be used if passed instead of creating or inferring one internally. |

### Methods

#### `attack`

```python
attack(X: numpy.ndarray, Y: numpy.ndarray, verbose: bool = False) -> Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to attack. |
| `Y` | `numpy.ndarray` |  | Labels for `X`. |
| `verbose` | `bool` | `False` | Print intermediate results of optimization if ``True``. |

**Returns**
- Type: `Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`

#### `explain`

```python
explain(X: numpy.ndarray, Y: Optional[numpy.ndarray] = None, verbose: bool = False) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances to attack. |
| `Y` | `Optional[numpy.ndarray]` | `None` | Labels for `X`. |
| `verbose` | `bool` | `False` | Print intermediate results of optimization if ``True``. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, no_info_type: str = 'median') -> alibi.explainers.cem.CEM
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  | Representative sample from the training data. |
| `no_info_type` | `str` | `'median'` | Median or mean value by feature supported. |

**Returns**
- Type: `alibi.explainers.cem.CEM`

#### `get_gradients`

```python
get_gradients(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance around which gradient is evaluated. |
| `Y` | `numpy.ndarray` |  | One-hot representation of instance labels. |

**Returns**
- Type: `numpy.ndarray`

#### `loss_fn`

```python
loss_fn(pred_proba: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `pred_proba` | `numpy.ndarray` |  | Prediction probabilities of an instance. |
| `Y` | `numpy.ndarray` |  | One-hot representation of instance labels. |

**Returns**
- Type: `numpy.ndarray`

#### `perturb`

```python
perturb(X: numpy.ndarray, eps: Union[float, numpy.ndarray], proba: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array to be perturbed. |
| `eps` | `Union[float, numpy.ndarray]` |  | Size of perturbation. |
| `proba` | `bool` | `False` | If ``True``, the net effect of the perturbation needs to be 0 to keep the sum of the probabilities equal to 1. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  | New predictor function/model. |

**Returns**
- Type: `None`
