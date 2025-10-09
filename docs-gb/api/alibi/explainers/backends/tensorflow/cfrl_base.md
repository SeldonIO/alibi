# `alibi.explainers.backends.tensorflow.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base`, for the Tensorflow backend.

## `TfCounterfactualRLDataset`

_Inherits from:_ `CounterfactualRLDataset`, `ABC`, `PyDataset`

Tensorflow backend datasets.

### Constructor

```python
TfCounterfactualRLDataset(self, X: numpy.ndarray, preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool = True) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `preprocessor` | `Callable` |  |  |
| `predictor` | `Callable` |  |  |
| `conditional_func` | `Callable` |  |  |
| `batch_size` | `int` |  |  |
| `shuffle` | `bool` | `True` |  |

### Methods

#### `on_epoch_end`

```python
on_epoch_end() -> None
```

This method is called every epoch and performs dataset shuffling.

**Returns**
- Type: `None`

## Functions
### `add_noise`

```python
add_noise(Z_cf: Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray], noise: NormalActionNoise, act_low: float, act_high: float, step: int, exploration_steps: int, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

Add noise to the counterfactual embedding.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  |  |
| `noise` | `NormalActionNoise` |  |  |
| `act_low` | `float` |  |  |
| `act_high` | `float` |  |  |
| `step` | `int` |  |  |
| `exploration_steps` | `int` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `consistency_loss`

```python
consistency_loss(Z_cf_pred: tensorflow.python.framework.tensor.Tensor, Z_cf_tgt: tensorflow.python.framework.tensor.Tensor)
```

Default 0 consistency loss.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `Z_cf_tgt` | `tensorflow.python.framework.tensor.Tensor` |  |  |

### `data_generator`

```python
data_generator(X: numpy.ndarray, encoder_preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool = True, kwargs)
```

Constructs a `tensorflow` data generator.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `encoder_preprocessor` | `Callable` |  |  |
| `predictor` | `Callable` |  |  |
| `conditional_func` | `Callable` |  |  |
| `batch_size` | `int` |  |  |
| `shuffle` | `bool` | `True` |  |

### `decode`

```python
decode(Z: Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray], decoder: keras.src.models.model.Model, kwargs)
```

Decodes an embedding tensor.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  |  |
| `decoder` | `keras.src.models.model.Model` |  |  |

### `encode`

```python
encode(X: Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray], encoder: keras.src.models.model.Model, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

Encodes the input tensor.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  |  |
| `encoder` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `generate_cf`

```python
generate_cf(Z: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], Y_m: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], Y_t: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], C: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None], actor: keras.src.models.model.Model, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

Generates counterfactual embedding.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |
| `Y_m` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |
| `Y_t` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |
| `C` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]` |  |  |
| `actor` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `get_actor`

```python
get_actor(hidden_dim: int, output_dim: int) -> keras.src.layers.layer.Layer
```

Constructs the actor network.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |
| `output_dim` | `int` |  |  |

**Returns**
- Type: `keras.src.layers.layer.Layer`

### `get_critic`

```python
get_critic(hidden_dim: int) -> keras.src.layers.layer.Layer
```

Constructs the critic network.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |

**Returns**
- Type: `keras.src.layers.layer.Layer`

### `get_optimizer`

```python
get_optimizer(model: Optional[keras.src.layers.layer.Layer] = None, lr: float = 0.001) -> keras.src.optimizers.optimizer.Optimizer
```

Constructs default `Adam` optimizer.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Optional[keras.src.layers.layer.Layer]` | `None` |  |
| `lr` | `float` | `0.001` |  |

**Returns**
- Type: `keras.src.optimizers.optimizer.Optimizer`

### `initialize_actor_critic`

```python
initialize_actor_critic(actor, critic, Z, Z_cf_tilde, Y_m, Y_t, C, kwargs)
```

Initialize actor and critic layers by passing a dummy zero tensor.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `actor` |  |  |  |
| `critic` |  |  |  |
| `Z` |  |  |  |
| `Z_cf_tilde` |  |  |  |
| `Y_m` |  |  |  |
| `Y_t` |  |  |  |
| `C` |  |  |  |

### `initialize_optimizer`

```python
initialize_optimizer(optimizer: keras.src.optimizers.optimizer.Optimizer, model: keras.src.models.model.Model) -> None
```

Initializes an optimizer given a model.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `keras.src.optimizers.optimizer.Optimizer` |  |  |
| `model` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `None`

### `initialize_optimizers`

```python
initialize_optimizers(optimizer_actor, optimizer_critic, actor, critic, kwargs) -> None
```

Initializes the actor and critic optimizers.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer_actor` |  |  |  |
| `optimizer_critic` |  |  |  |
| `actor` |  |  |  |
| `critic` |  |  |  |

**Returns**
- Type: `None`

### `load_model`

```python
load_model(path: Union[str, os.PathLike]) -> keras.src.models.model.Model
```

Loads a model and its optimizer.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `keras.src.models.model.Model`

### `save_model`

```python
save_model(path: Union[str, os.PathLike], model: keras.src.layers.layer.Layer) -> None
```

Saves a model and its optimizer.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |
| `model` | `keras.src.layers.layer.Layer` |  |  |

**Returns**
- Type: `None`

### `set_seed`

```python
set_seed(seed: int = 13)
```

Sets a seed to ensure reproducibility. Does NOT ensure reproducibility.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `seed` | `int` | `13` |  |

### `sparsity_loss`

```python
sparsity_loss(X_hat_cf: tensorflow.python.framework.tensor.Tensor, X: tensorflow.python.framework.tensor.Tensor) -> Dict[str, tensorflow.python.framework.tensor.Tensor]
```

Default L1 sparsity loss.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cf` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `X` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Dict[str, tensorflow.python.framework.tensor.Tensor]`

### `to_numpy`

```python
to_numpy(X: Union[List[Any], numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]) -> Union[List[Any], numpy.ndarray, None]
```

Converts given tensor to `numpy` array.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[List[Any], numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]` |  |  |

**Returns**
- Type: `Union[List[Any], numpy.ndarray, None]`

### `to_tensor`

```python
to_tensor(X: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], kwargs) -> Optional[tensorflow.python.framework.tensor.Tensor]
```

Converts tensor to `tf.Tensor`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |

**Returns**
- Type: `Optional[tensorflow.python.framework.tensor.Tensor]`
