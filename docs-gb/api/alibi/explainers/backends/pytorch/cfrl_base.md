# `alibi.explainers.backends.pytorch.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base` for the Pytorch backend.

## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
## `PtCounterfactualRLDataset`

_Inherits from:_ `CounterfactualRLDataset`, `ABC`, `Dataset`, `Generic`

Pytorch backend datasets.

### Constructor

```python
PtCounterfactualRLDataset(self, X: numpy.ndarray, preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `preprocessor` | `Callable` |  |  |
| `predictor` | `Callable` |  |  |
| `conditional_func` | `Callable` |  |  |
| `batch_size` | `int` |  |  |

## Functions
### `add_noise`

```python
add_noise(Z_cf: torch.Tensor, noise: NormalActionNoise, act_low: float, act_high: float, step: int, exploration_steps: int, device: torch.device, kwargs) -> torch.Tensor
```

Add noise to the counterfactual embedding.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf` | `torch.Tensor` |  |  |
| `noise` | `NormalActionNoise` |  |  |
| `act_low` | `float` |  |  |
| `act_high` | `float` |  |  |
| `step` | `int` |  |  |
| `exploration_steps` | `int` |  |  |
| `device` | `torch.device` |  |  |

**Returns**
- Type: `torch.Tensor`

### `consistency_loss`

```python
consistency_loss(Z_cf_pred: torch.Tensor, Z_cf_tgt: torch.Tensor)
```

Default 0 consistency loss.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `torch.Tensor` |  |  |
| `Z_cf_tgt` | `torch.Tensor` |  |  |

### `data_generator`

```python
data_generator(X: numpy.ndarray, encoder_preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool, num_workers: int, kwargs)
```

Constructs a tensorflow data generator.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `encoder_preprocessor` | `Callable` |  |  |
| `predictor` | `Callable` |  |  |
| `conditional_func` | `Callable` |  |  |
| `batch_size` | `int` |  |  |
| `shuffle` | `bool` |  |  |
| `num_workers` | `int` |  |  |

### `decode`

```python
decode(Z: torch.Tensor, decoder: torch.nn.modules.module.Module, device: torch.device, kwargs)
```

Decodes an embedding tensor.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `torch.Tensor` |  |  |
| `decoder` | `torch.nn.modules.module.Module` |  |  |
| `device` | `torch.device` |  |  |

### `encode`

```python
encode(X: torch.Tensor, encoder: torch.nn.modules.module.Module, device: torch.device, kwargs)
```

Encodes the input tensor.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `torch.Tensor` |  |  |
| `encoder` | `torch.nn.modules.module.Module` |  |  |
| `device` | `torch.device` |  |  |

### `generate_cf`

```python
generate_cf(Z: torch.Tensor, Y_m: torch.Tensor, Y_t: torch.Tensor, C: Optional[torch.Tensor], encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, actor: torch.nn.modules.module.Module, device: torch.device, kwargs) -> torch.Tensor
```

Generates counterfactual embedding.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `torch.Tensor` |  |  |
| `Y_m` | `torch.Tensor` |  |  |
| `Y_t` | `torch.Tensor` |  |  |
| `C` | `Optional[torch.Tensor]` |  |  |
| `encoder` | `torch.nn.modules.module.Module` |  |  |
| `decoder` | `torch.nn.modules.module.Module` |  |  |
| `actor` | `torch.nn.modules.module.Module` |  |  |
| `device` | `torch.device` |  |  |

**Returns**
- Type: `torch.Tensor`

### `get_actor`

```python
get_actor(hidden_dim: int, output_dim: int) -> torch.nn.modules.module.Module
```

Constructs the actor network.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |
| `output_dim` | `int` |  |  |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `get_critic`

```python
get_critic(hidden_dim: int) -> torch.nn.modules.module.Module
```

Constructs the critic network.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `get_device`

```python
get_device() -> torch.device
```

Checks if `cuda` is available. If available, use `cuda` by default, else use `cpu`.

Returns

**Returns**
- Type: `torch.device`

### `get_optimizer`

```python
get_optimizer(model: torch.nn.modules.module.Module, lr: float = 0.001) -> torch.optim.optimizer.Optimizer
```

Constructs default `Adam` optimizer.

Returns

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `torch.nn.modules.module.Module` |  |  |
| `lr` | `float` | `0.001` |  |

**Returns**
- Type: `torch.optim.optimizer.Optimizer`

### `load_model`

```python
load_model(path: Union[str, os.PathLike]) -> torch.nn.modules.module.Module
```

Loads a model and its optimizer.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `save_model`

```python
save_model(path: Union[str, os.PathLike], model: torch.nn.modules.module.Module) -> None
```

Saves a model and its optimizer.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |
| `model` | `torch.nn.modules.module.Module` |  |  |

**Returns**
- Type: `None`

### `set_seed`

```python
set_seed(seed: int = 13)
```

Sets a seed to ensure reproducibility.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `seed` | `int` | `13` |  |

### `sparsity_loss`

```python
sparsity_loss(X_hat_cf: torch.Tensor, X: torch.Tensor) -> Dict[str, torch.Tensor]
```

Default L1 sparsity loss.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cf` | `torch.Tensor` |  |  |
| `X` | `torch.Tensor` |  |  |

**Returns**
- Type: `Dict[str, torch.Tensor]`

### `to_numpy`

```python
to_numpy(X: Union[List[Any], numpy.ndarray, torch.Tensor, None]) -> Union[List[Any], numpy.ndarray, None]
```

Converts given tensor to `numpy` array.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[List[Any], numpy.ndarray, torch.Tensor, None]` |  |  |

**Returns**
- Type: `Union[List[Any], numpy.ndarray, None]`

### `to_tensor`

```python
to_tensor(X: Union[numpy.ndarray, torch.Tensor], device: torch.device, kwargs) -> Optional[torch.Tensor]
```

Converts tensor to `torch.Tensor`

Returns

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, torch.Tensor]` |  |  |
| `device` | `torch.device` |  |  |

**Returns**
- Type: `Optional[torch.Tensor]`

### `update_actor_critic`

```python
update_actor_critic(encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, critic: torch.nn.modules.module.Module, actor: torch.nn.modules.module.Module, optimizer_critic: torch.optim.optimizer.Optimizer, optimizer_actor: torch.optim.optimizer.Optimizer, sparsity_loss: Callable, consistency_loss: Callable, coeff_sparsity: float, coeff_consistency: float, X: numpy.ndarray, X_cf: numpy.ndarray, Z: numpy.ndarray, Z_cf_tilde: numpy.ndarray, Y_m: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[numpy.ndarray], R_tilde: numpy.ndarray, device: torch.device, kwargs)
```

Training step. Updates actor and critic networks including additional losses.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `torch.nn.modules.module.Module` |  |  |
| `decoder` | `torch.nn.modules.module.Module` |  |  |
| `critic` | `torch.nn.modules.module.Module` |  |  |
| `actor` | `torch.nn.modules.module.Module` |  |  |
| `optimizer_critic` | `torch.optim.optimizer.Optimizer` |  |  |
| `optimizer_actor` | `torch.optim.optimizer.Optimizer` |  |  |
| `sparsity_loss` | `Callable` |  |  |
| `consistency_loss` | `Callable` |  |  |
| `coeff_sparsity` | `float` |  |  |
| `coeff_consistency` | `float` |  |  |
| `X` | `numpy.ndarray` |  |  |
| `X_cf` | `numpy.ndarray` |  |  |
| `Z` | `numpy.ndarray` |  |  |
| `Z_cf_tilde` | `numpy.ndarray` |  |  |
| `Y_m` | `numpy.ndarray` |  |  |
| `Y_t` | `numpy.ndarray` |  |  |
| `C` | `Optional[numpy.ndarray]` |  |  |
| `R_tilde` | `numpy.ndarray` |  |  |
| `device` | `torch.device` |  |  |
