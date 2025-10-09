# `alibi.explainers.cem`
## `CEM`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
CEM(self, predict: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], mode: str, shape: tuple, kappa: float = 0.0, beta: float = 0.1, feature_range: tuple = (-10000000000.0, 10000000000.0), gamma: float = 0.0, ae_model: Optional[keras.src.models.model.Model] = None, learning_rate_init: float = 0.01, max_iterations: int = 1000, c_init: float = 10.0, c_steps: int = 10, eps: tuple = (0.001, 0.001), clip: tuple = (-100.0, 100.0), update_num_grad: int = 1, no_info_val: Union[float, numpy.ndarray, NoneType] = None, write_dir: Optional[str] = None, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  |  |
| `mode` | `str` |  |  |
| `shape` | `tuple` |  |  |
| `kappa` | `float` | `0.0` |  |
| `beta` | `float` | `0.1` |  |
| `feature_range` | `tuple` | `(-10000000000.0, 10000000000.0)` |  |
| `gamma` | `float` | `0.0` |  |
| `ae_model` | `Optional[keras.src.models.model.Model]` | `None` |  |
| `learning_rate_init` | `float` | `0.01` |  |
| `max_iterations` | `int` | `1000` |  |
| `c_init` | `float` | `10.0` |  |
| `c_steps` | `int` | `10` |  |
| `eps` | `tuple` | `(0.001, 0.001)` |  |
| `clip` | `tuple` | `(-100.0, 100.0)` |  |
| `update_num_grad` | `int` | `1` |  |
| `no_info_val` | `Union[float, numpy.ndarray, None]` | `None` |  |
| `write_dir` | `Optional[str]` | `None` |  |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` |  |

### Methods

#### `attack`

```python
attack(X: numpy.ndarray, Y: numpy.ndarray, verbose: bool = False) -> Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
```

Find pertinent negative or pertinent positive for instance `X` using a fast iterative

shrinkage-thresholding algorithm (FISTA).

Parameters
----------
X
    Instance to attack.
Y
    Labels for `X`.
verbose
    Print intermediate results of optimization if ``True``.

Returns
-------
Overall best attack and gradients for that attack.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |
| `verbose` | `bool` | `False` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`

#### `explain`

```python
explain(X: numpy.ndarray, Y: Optional[numpy.ndarray] = None, verbose: bool = False) -> alibi.api.interfaces.Explanation
```

Explain instance and return PP or PN with metadata.

Parameters
----------
X
    Instances to attack.
Y
    Labels for `X`.
verbose
    Print intermediate results of optimization if ``True``.

Returns
-------
explanation
    `Explanation` object containing the PP or PN with additional metadata as attributes.
    See usage at `CEM examples`_ for details.

    .. _CEM examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/CEM.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `Optional[numpy.ndarray]` | `None` |  |
| `verbose` | `bool` | `False` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, no_info_type: str = 'median') -> alibi.explainers.cem.CEM
```

Get 'no information' values from the training data.

Parameters
----------
train_data
    Representative sample from the training data.
no_info_type
    Median or mean value by feature supported.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  |  |
| `no_info_type` | `str` | `'median'` |  |

**Returns**
- Type: `alibi.explainers.cem.CEM`

#### `get_gradients`

```python
get_gradients(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

Compute numerical gradients of the attack loss term:

`dL/dx = (dL/dP)*(dP/dx)` with `L = loss_attack_s; P = predict; x = adv_s`

Parameters
----------
X
    Instance around which gradient is evaluated.
Y
    One-hot representation of instance labels.

Returns
-------
Array with gradients.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `loss_fn`

```python
loss_fn(pred_proba: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

Compute the attack loss.

Parameters
----------
pred_proba
    Prediction probabilities of an instance.
Y
    One-hot representation of instance labels.

Returns
-------
Loss of the attack.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `pred_proba` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `perturb`

```python
perturb(X: numpy.ndarray, eps: Union[float, numpy.ndarray], proba: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Apply perturbation to instance or prediction probabilities. Used for numerical calculation of gradients.

Parameters
----------
X
    Array to be perturbed.
eps
    Size of perturbation.
proba
    If ``True``, the net effect of the perturbation needs to be 0 to keep the sum of the
    probabilities equal to 1.

Returns
-------
Instances where a positive and negative perturbation is applied.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `eps` | `Union[float, numpy.ndarray]` |  |  |
| `proba` | `bool` | `False` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

Resets the predictor function/model.

Parameters
----------
predictor
    New predictor function/model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  |  |

**Returns**
- Type: `None`
