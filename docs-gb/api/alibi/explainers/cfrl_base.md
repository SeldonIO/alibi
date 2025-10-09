# `alibi.explainers.cfrl_base`
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
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `encoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `decoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `coeff_sparsity` | `float` |  |  |
| `coeff_consistency` | `float` |  |  |
| `latent_dim` | `Optional[int]` | `None` |  |
| `backend` | `str` | `'tensorflow'` |  |
| `seed` | `int` | `0` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[numpy.ndarray] = None, batch_size: int = 100) -> alibi.api.interfaces.Explanation
```

Explains an input instance

Parameters
----------
X
    Instances to be explained.
Y_t
    Counterfactual targets.
C
    Conditional vectors. If ``None``, it means that no conditioning was used during training (i.e. the
    `conditional_func` returns ``None``).
batch_size
    Batch size to be used when generating counterfactuals.

Returns
-------
explanation
    `Explanation` object containing the counterfactual with additional metadata as attributes.             See usage at `CFRL examples`_ for details.

    .. _CFRL examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y_t` | `numpy.ndarray` |  |  |
| `C` | `Optional[numpy.ndarray]` | `None` |  |
| `batch_size` | `int` | `100` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(X: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

Fit the model agnostic counterfactual generator.

Parameters
----------
X
    Training data array.

Returns
-------
self
    The explainer itself.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

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
| `mu` | `float` |  |  |
| `sigma` | `float` |  |  |

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
| `size` | `int` | `1000` |  |

### Methods

#### `append`

```python
append(X: numpy.ndarray, Y_m: numpy.ndarray, Y_t: numpy.ndarray, Z: numpy.ndarray, Z_cf_tilde: numpy.ndarray, C: Optional[numpy.ndarray], R_tilde: numpy.ndarray, kwargs) -> None
```

Adds experience to the replay buffer. When the buffer is filled, then the oldest experience is replaced

by the new one (FIFO).

Parameters
----------
X
    Input array.
Y_m
    Model's prediction class of `X`.
Y_t
    Counterfactual target class.
Z
    Input's embedding.
Z_cf_tilde
    Noised counterfactual embedding.
C
    Conditional array.
R_tilde
    Noised counterfactual reward array.
**kwargs
    Other arguments. Not used.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y_m` | `numpy.ndarray` |  |  |
| `Y_t` | `numpy.ndarray` |  |  |
| `Z` | `numpy.ndarray` |  |  |
| `Z_cf_tilde` | `numpy.ndarray` |  |  |
| `C` | `Optional[numpy.ndarray]` |  |  |
| `R_tilde` | `numpy.ndarray` |  |  |

**Returns**
- Type: `None`

#### `sample`

```python
sample() -> Dict[str, Optional[numpy.ndarray]]
```

Sample a batch of experience form the replay buffer.

Returns
-------
A batch experience. For a description of the keys and values returned, see parameter descriptions         in :py:meth:`alibi.explainers.cfrl_base.ReplayBuffer.append` method. The batch size returned is the same         as the one passed in the :py:meth:`alibi.explainers.cfrl_base.ReplayBuffer.append`.

**Returns**
- Type: `Dict[str, Optional[numpy.ndarray]]`
