# `alibi.explainers.backends.pytorch.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base` for the Pytorch backend.

## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

## `PtCounterfactualRLDataset`

_Inherits from:_ `CounterfactualRLDataset`, `ABC`, `Dataset`, `Generic`

Pytorch backend datasets.

### Constructor

```python
PtCounterfactualRLDataset(self, X: numpy.ndarray, preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling the `preprocessor` function. |
| `preprocessor` | `Callable` |  | Preprocessor function. This function correspond to the preprocessing steps applied to the auto-encoder model. |
| `predictor` | `Callable` |  | Prediction function. The classifier function should expect the input in the original format and preprocess it internally in the `predictor` if necessary. |
| `conditional_func` | `Callable` |  | Conditional function generator. Given an preprocessed input array, the functions generates a conditional array. |
| `batch_size` | `int` |  | Dimension of the batch used during training. The same batch size is used to infer the classification labels of the input dataset. |

## Functions
### `add_noise`

```python
add_noise(Z_cf: torch.Tensor, noise: NormalActionNoise, act_low: float, act_high: float, step: int, exploration_steps: int, device: torch.device, kwargs) -> torch.Tensor
```

Add noise to the counterfactual embedding.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf` | `torch.Tensor` |  | Counterfactual embedding. |
| `noise` | `NormalActionNoise` |  | Noise generator object. |
| `act_low` | `float` |  | Action lower bound. |
| `act_high` | `float` |  | Action upper bound. |
| `step` | `int` |  | Training step. |
| `exploration_steps` | `int` |  | Number of exploration steps. For the first `exploration_steps`, the noised counterfactual embedding is sampled uniformly at random. |
| `device` | `torch.device` |  | Device to send data to. |

**Returns**
- Type: `torch.Tensor`

### `consistency_loss`

```python
consistency_loss(Z_cf_pred: torch.Tensor, Z_cf_tgt: torch.Tensor)
```

Default 0 consistency loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `torch.Tensor` |  | Counterfactual embedding prediction. |
| `Z_cf_tgt` | `torch.Tensor` |  | Counterfactual embedding target. |

### `data_generator`

```python
data_generator(X: numpy.ndarray, encoder_preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool, num_workers: int, kwargs)
```

Constructs a tensorflow data generator.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling the `preprocessor` function. |
| `encoder_preprocessor` | `Callable` |  | Preprocessor function. This function correspond to the preprocessing steps applied to the encoder/auto-encoder model. |
| `predictor` | `Callable` |  | Prediction function. The classifier function should expect the input in the original format and preprocess it internally in the `predictor` if necessary. |
| `conditional_func` | `Callable` |  | Conditional function generator. Given an preprocessed input array, the functions generates a conditional array. |
| `batch_size` | `int` |  | Dimension of the batch used during training. The same batch size is used to infer the classification labels of the input dataset. |
| `shuffle` | `bool` |  | Whether to shuffle the dataset each epoch. ``True`` by default. |
| `num_workers` | `int` |  | Number of worker processes to be created. |
| `**kwargs` |  |  | Other arguments. Not used. |

### `decode`

```python
decode(Z: torch.Tensor, decoder: torch.nn.modules.module.Module, device: torch.device, kwargs)
```

Decodes an embedding tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `torch.Tensor` |  | Embedding tensor to be decoded. |
| `decoder` | `torch.nn.modules.module.Module` |  | Pretrained decoder network. |
| `device` | `torch.device` |  | Device to sent data to. |

### `encode`

```python
encode(X: torch.Tensor, encoder: torch.nn.modules.module.Module, device: torch.device, kwargs)
```

Encodes the input tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `torch.Tensor` |  | Input to be encoded. |
| `encoder` | `torch.nn.modules.module.Module` |  | Pretrained encoder network. |
| `device` | `torch.device` |  | Device to send data to. |

### `generate_cf`

```python
generate_cf(Z: torch.Tensor, Y_m: torch.Tensor, Y_t: torch.Tensor, C: Optional[torch.Tensor], encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, actor: torch.nn.modules.module.Module, device: torch.device, kwargs) -> torch.Tensor
```

Generates counterfactual embedding.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `torch.Tensor` |  | Input embedding tensor. |
| `Y_m` | `torch.Tensor` |  | Input classification label. |
| `Y_t` | `torch.Tensor` |  | Target counterfactual classification label. |
| `C` | `Optional[torch.Tensor]` |  | Conditional tensor. |
| `encoder` | `torch.nn.modules.module.Module` |  | Pretrained encoder network. |
| `decoder` | `torch.nn.modules.module.Module` |  | Pretrained decoder network. |
| `actor` | `torch.nn.modules.module.Module` |  | Actor network. The model generates the counterfactual embedding. |
| `device` | `torch.device` |  | Device object to be used. |

**Returns**
- Type: `torch.Tensor`

### `get_actor`

```python
get_actor(hidden_dim: int, output_dim: int) -> torch.nn.modules.module.Module
```

Constructs the actor network.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Actor's hidden dimension |
| `output_dim` | `int` |  | Actor's output dimension. |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `get_critic`

```python
get_critic(hidden_dim: int) -> torch.nn.modules.module.Module
```

Constructs the critic network.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Critic's hidden dimension. |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `get_device`

```python
get_device() -> torch.device
```

Checks if `cuda` is available. If available, use `cuda` by default, else use `cpu`.

**Returns**
- Type: `torch.device`

### `get_optimizer`

```python
get_optimizer(model: torch.nn.modules.module.Module, lr: float = 0.001) -> torch.optim.optimizer.Optimizer
```

Constructs default `Adam` optimizer.

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to the loading location. |

**Returns**
- Type: `torch.nn.modules.module.Module`

### `save_model`

```python
save_model(path: Union[str, os.PathLike], model: torch.nn.modules.module.Module) -> None
```

Saves a model and its optimizer.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to the saving location. |
| `model` | `torch.nn.modules.module.Module` |  | Model to be saved. |

**Returns**
- Type: `None`

### `set_seed`

```python
set_seed(seed: int = 13)
```

Sets a seed to ensure reproducibility.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `seed` | `int` | `13` | Seed to be set. |

### `sparsity_loss`

```python
sparsity_loss(X_hat_cf: torch.Tensor, X: torch.Tensor) -> Dict[str, torch.Tensor]
```

Default L1 sparsity loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cf` | `torch.Tensor` |  | Auto-encoder counterfactual reconstruction. |
| `X` | `torch.Tensor` |  | Input instance |

**Returns**
- Type: `Dict[str, torch.Tensor]`

### `to_numpy`

```python
to_numpy(X: Union[List[Any], numpy.ndarray, torch.Tensor, None]) -> Union[List[Any], numpy.ndarray, None]
```

Converts given tensor to `numpy` array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[List[Any], numpy.ndarray, torch.Tensor, None]` |  | Input tensor to be converted to `numpy` array. |

**Returns**
- Type: `Union[List[Any], numpy.ndarray, None]`

### `to_tensor`

```python
to_tensor(X: Union[numpy.ndarray, torch.Tensor], device: torch.device, kwargs) -> Optional[torch.Tensor]
```

Converts tensor to `torch.Tensor`

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `torch.nn.modules.module.Module` |  | Pretrained encoder network. |
| `decoder` | `torch.nn.modules.module.Module` |  | Pretrained decoder network. |
| `critic` | `torch.nn.modules.module.Module` |  | Critic network. |
| `actor` | `torch.nn.modules.module.Module` |  | Actor network. |
| `optimizer_critic` | `torch.optim.optimizer.Optimizer` |  | Critic's optimizer. |
| `optimizer_actor` | `torch.optim.optimizer.Optimizer` |  | Actor's optimizer. |
| `sparsity_loss` | `Callable` |  | Sparsity loss function. |
| `consistency_loss` | `Callable` |  | Consistency loss function. |
| `coeff_sparsity` | `float` |  | Sparsity loss coefficient. |
| `coeff_consistency` | `float` |  | Consistency loss coefficient |
| `X` | `numpy.ndarray` |  | Input array. |
| `X_cf` | `numpy.ndarray` |  | Counterfactual array. |
| `Z` | `numpy.ndarray` |  | Input embedding. |
| `Z_cf_tilde` | `numpy.ndarray` |  | Noised counterfactual embedding. |
| `Y_m` | `numpy.ndarray` |  | Input classification label. |
| `Y_t` | `numpy.ndarray` |  | Target counterfactual classification label. |
| `C` | `Optional[numpy.ndarray]` |  | Conditional tensor. |
| `R_tilde` | `numpy.ndarray` |  | Noised counterfactual reward. |
| `device` | `torch.device` |  | Torch device object. |
| `**kwargs` |  |  | Other arguments. Not used. |
