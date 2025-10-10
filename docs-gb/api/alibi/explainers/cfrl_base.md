# `alibi.explainers.cfrl_base`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `DEFAULT_DATA_CFRL`
```python
DEFAULT_DATA_CFRL: dict = {'orig': None, 'cf': None, 'target': None, 'condition': None}
```

### `DEFAULT_META_CFRL`
```python
DEFAULT_META_CFRL: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.cfrl_base (WARNING)>
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

### `DEFAULT_BASE_PARAMS`
```python
DEFAULT_BASE_PARAMS: dict = {'act_noise': 0.1, 'act_low': -1.0, 'act_high': 1.0, 'replay_buffer_size': 10...
```

## `Callback`

_Inherits from:_ `ABC`

Training callback class.

## `CounterfactualRL`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

Counterfactual Reinforcement Learning.

### Constructor

```python
CounterfactualRL(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], encoder: 'Union[tensorflow.keras.Model, torch.nn.Module]', decoder: 'Union[tensorflow.keras.Model, torch.nn.Module]', coeff_sparsity: float, coeff_consistency: float, latent_dim: Optional[int] = None, backend: str = 'tensorflow', seed: int = 0, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. For classification task, the second dimension of the output should match the number of classes. Thus, the output can be either a soft label distribution or a hard label distribution (i.e. one-hot encoding) without affecting the performance since `argmax` is applied to the predictor's output. |
| `encoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Pretrained encoder network. |
| `decoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Pretrained decoder network. |
| `coeff_sparsity` | `float` |  | Sparsity loss coefficient. |
| `coeff_consistency` | `float` |  | Consistency loss coefficient. |
| `latent_dim` | `Optional[int]` | `None` | Auto-encoder latent dimension. Can be omitted if the actor network is user specified. |
| `backend` | `str` | `'tensorflow'` | Deep learning backend: ``'tensorflow'`` | ``'pytorch'``. Default ``'tensorflow'``. |
| `seed` | `int` | `0` | Seed for reproducibility. The results are not reproducible for ``'tensorflow'`` backend. |
| `**kwargs` |  |  | Used to replace any default parameter from :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS`. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[numpy.ndarray] = None, batch_size: int = 100) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances to be explained. |
| `Y_t` | `numpy.ndarray` |  | Counterfactual targets. |
| `C` | `Optional[numpy.ndarray]` | `None` | Conditional vectors. If ``None``, it means that no conditioning was used during training (i.e. the `conditional_func` returns ``None``). |
| `batch_size` | `int` | `100` | Batch size to be used when generating counterfactuals. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(X: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training data array. |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

#### `load`

```python
load(path: Union[str, os.PathLike], predictor: typing.Any) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |
| `predictor` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

#### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  |  |

**Returns**
- Type: `None`

#### `save`

```python
save(path: Union[str, os.PathLike]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

## `NormalActionNoise`

Normal noise generator.

### Constructor

```python
NormalActionNoise(self, mu: float, sigma: float) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `mu` | `float` |  | Mean of the normal noise. |
| `sigma` | `float` |  | Standard deviation of the noise. |

## `Postprocessing`

_Inherits from:_ `ABC`

## `ReplayBuffer`

Circular experience replay buffer for `CounterfactualRL` (DDPG). When the buffer is filled, then the oldest
experience is replaced by the new one (FIFO). The experience batch size is kept constant and inferred when
the first batch of data is stored. Allowing flexible batch size can generate `tensorflow` warning due to
the `tf.function` retracing, which can lead to a drop in performance.

### Constructor

```python
ReplayBuffer(self, size: int = 1000) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `size` | `int` | `1000` | Dimension of the buffer in batch size. This that the total memory allocated is proportional with the `size x batch_size`, where `batch_size` is inferred from the first array to be stored. |

### Methods

#### `append`

```python
append(X: numpy.ndarray, Y_m: numpy.ndarray, Y_t: numpy.ndarray, Z: numpy.ndarray, Z_cf_tilde: numpy.ndarray, C: Optional[numpy.ndarray], R_tilde: numpy.ndarray, kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Input array. |
| `Y_m` | `numpy.ndarray` |  | Model's prediction class of `X`. |
| `Y_t` | `numpy.ndarray` |  | Counterfactual target class. |
| `Z` | `numpy.ndarray` |  | Input's embedding. |
| `Z_cf_tilde` | `numpy.ndarray` |  | Noised counterfactual embedding. |
| `C` | `Optional[numpy.ndarray]` |  | Conditional array. |
| `R_tilde` | `numpy.ndarray` |  | Noised counterfactual reward array. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `None`

#### `sample`

```python
sample() -> Dict[str, Optional[numpy.ndarray]]
```

Sample a batch of experience form the replay buffer.

**Returns**
- Type: `Dict[str, Optional[numpy.ndarray]]`
