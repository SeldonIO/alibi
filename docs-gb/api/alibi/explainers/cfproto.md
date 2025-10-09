# `alibi.explainers.cfproto`
## Constants
### `DEFAULT_DATA_CFP`
```python
DEFAULT_DATA_CFP: dict = {'all': [], 'cf': None, 'id_proto': None, 'orig_class': None, 'orig_proba': None}
```
### `DEFAULT_META_CFP`
```python
DEFAULT_META_CFP: dict = { 'explanations': ['local'],
  'name': None,
  'params': {},
  'type': ['blackbox', 'tensorflow', 'keras'],
  'version': None}
```
### `logger`
```python
logger: Logger = <Logger alibi.explainers.cfproto (WARNING)>
```
## `CounterfactualProto`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
CounterfactualProto(self, predict: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], shape: tuple, kappa: float = 0.0, beta: float = 0.1, feature_range: Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]] = (-10000000000.0, 10000000000.0), gamma: float = 0.0, ae_model: Optional[keras.src.models.model.Model] = None, enc_model: Optional[keras.src.models.model.Model] = None, theta: float = 0.0, cat_vars: Optional[Dict[int, int]] = None, ohe: bool = False, use_kdtree: bool = False, learning_rate_init: float = 0.01, max_iterations: int = 1000, c_init: float = 10.0, c_steps: int = 10, eps: tuple = (0.001, 0.001), clip: tuple = (-1000.0, 1000.0), update_num_grad: int = 1, write_dir: Optional[str] = None, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  |  |
| `shape` | `tuple` |  |  |
| `kappa` | `float` | `0.0` |  |
| `beta` | `float` | `0.1` |  |
| `feature_range` | `Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]]` | `(-10000000000.0, 10000000000.0)` |  |
| `gamma` | `float` | `0.0` |  |
| `ae_model` | `Optional[keras.src.models.model.Model]` | `None` |  |
| `enc_model` | `Optional[keras.src.models.model.Model]` | `None` |  |
| `theta` | `float` | `0.0` |  |
| `cat_vars` | `Optional[Dict[int, int]]` | `None` |  |
| `ohe` | `bool` | `False` |  |
| `use_kdtree` | `bool` | `False` |  |
| `learning_rate_init` | `float` | `0.01` |  |
| `max_iterations` | `int` | `1000` |  |
| `c_init` | `float` | `10.0` |  |
| `c_steps` | `int` | `10` |  |
| `eps` | `tuple` | `(0.001, 0.001)` |  |
| `clip` | `tuple` | `(-1000.0, 1000.0)` |  |
| `update_num_grad` | `int` | `1` |  |
| `write_dir` | `Optional[str]` | `None` |  |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` |  |

### Methods

#### `attack`

```python
attack(X: numpy.ndarray, Y: numpy.ndarray, target_class: Optional[list] = None, k: Optional[int] = None, k_type: str = 'mean', threshold: float = 0.0, verbose: bool = False, print_every: int = 100, log_every: int = 100) -> Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
```

Find a counterfactual (CF) for instance `X` using a fast iterative shrinkage-thresholding algorithm (FISTA).

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |
| `target_class` | `Optional[list]` | `None` |  |
| `k` | `Optional[int]` | `None` |  |
| `k_type` | `str` | `'mean'` |  |
| `threshold` | `float` | `0.0` |  |
| `verbose` | `bool` | `False` |  |
| `print_every` | `int` | `100` |  |
| `log_every` | `int` | `100` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`

#### `explain`

```python
explain(X: numpy.ndarray, Y: Optional[numpy.ndarray] = None, target_class: Optional[list] = None, k: Optional[int] = None, k_type: str = 'mean', threshold: float = 0.0, verbose: bool = False, print_every: int = 100, log_every: int = 100) -> alibi.api.interfaces.Explanation
```

Explain instance and return counterfactual with metadata.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `Optional[numpy.ndarray]` | `None` |  |
| `target_class` | `Optional[list]` | `None` |  |
| `k` | `Optional[int]` | `None` |  |
| `k_type` | `str` | `'mean'` |  |
| `threshold` | `float` | `0.0` |  |
| `verbose` | `bool` | `False` |  |
| `print_every` | `int` | `100` |  |
| `log_every` | `int` | `100` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, trustscore_kwargs: Optional[dict] = None, d_type: str = 'abdm', w: Optional[float] = None, disc_perc: Sequence[Union[int, float]] = (25, 50, 75), standardize_cat_vars: bool = False, smooth: float = 1.0, center: bool = True, update_feature_range: bool = True) -> alibi.explainers.cfproto.CounterfactualProto
```

Get prototypes for each class using the encoder or k-d trees.

The prototypes are used for the encoder loss term or to calculate the optional trust scores.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  |  |
| `trustscore_kwargs` | `Optional[dict]` | `None` |  |
| `d_type` | `str` | `'abdm'` |  |
| `w` | `Optional[float]` | `None` |  |
| `disc_perc` | `Sequence[Union[int, float]]` | `(25, 50, 75)` |  |
| `standardize_cat_vars` | `bool` | `False` |  |
| `smooth` | `float` | `1.0` |  |
| `center` | `bool` | `True` |  |
| `update_feature_range` | `bool` | `True` |  |

**Returns**
- Type: `alibi.explainers.cfproto.CounterfactualProto`

#### `get_gradients`

```python
get_gradients(X: numpy.ndarray, Y: numpy.ndarray, grads_shape: tuple, cat_vars_ord: dict) -> numpy.ndarray
```

Compute numerical gradients of the attack loss term:

`dL/dx = (dL/dP)*(dP/dx)` with `L = loss_attack_s; P = predict; x = adv_s`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |
| `grads_shape` | `tuple` |  |  |
| `cat_vars_ord` | `dict` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `loss_fn`

```python
loss_fn(pred_proba: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

Compute the attack loss.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `pred_proba` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

Resets the predictor function/model.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  |  |

**Returns**
- Type: `None`

#### `score`

```python
score(X: numpy.ndarray, adv_class: int, orig_class: int, eps: float = 1e-10) -> float
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `adv_class` | `int` |  |  |
| `orig_class` | `int` |  |  |
| `eps` | `float` | `1e-10` |  |

**Returns**
- Type: `float`

## Functions
### `CounterFactualProto`

```python
CounterFactualProto(args, kwargs)
```

The class name `CounterFactualProto` is deprecated, please use `CounterfactualProto`.
