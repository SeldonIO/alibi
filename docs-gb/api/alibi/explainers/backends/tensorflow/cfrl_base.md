# `alibi.explainers.backends.tensorflow.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base`, for the Tensorflow backend.

## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

## `TfCounterfactualRLDataset`

_Inherits from:_ `CounterfactualRLDataset`, `ABC`, `PyDataset`

Tensorflow backend datasets.

### Constructor

```python
TfCounterfactualRLDataset(self, X: numpy.ndarray, preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool = True) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling the `preprocessor` function. |
| `preprocessor` | `Callable` |  | Preprocessor function. This function correspond to the preprocessing steps applied to the encoder/auto-encoder model. |
| `predictor` | `Callable` |  | Prediction function. The classifier function should expect the input in the original format and preprocess it internally in the `predictor` if necessary. |
| `conditional_func` | `Callable` |  | Conditional function generator. Given an pre-processed input array, the functions generates a conditional array. |
| `batch_size` | `int` |  | Dimension of the batch used during training. The same batch size is used to infer the classification labels of the input dataset. |
| `shuffle` | `bool` | `True` | Whether to shuffle the dataset each epoch. ``True`` by default. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  | Counterfactual embedding. |
| `noise` | `NormalActionNoise` |  | Noise generator object. |
| `act_low` | `float` |  | Noise lower bound. |
| `act_high` | `float` |  | Noise upper bound. |
| `step` | `int` |  | Training step. |
| `exploration_steps` | `int` |  | Number of exploration steps. For the first `exploration_steps`, the noised counterfactual embedding is sampled uniformly at random. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `consistency_loss`

```python
consistency_loss(Z_cf_pred: tensorflow.python.framework.tensor.Tensor, Z_cf_tgt: tensorflow.python.framework.tensor.Tensor)
```

Default 0 consistency loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `tensorflow.python.framework.tensor.Tensor` |  | Counterfactual embedding prediction. |
| `Z_cf_tgt` | `tensorflow.python.framework.tensor.Tensor` |  | Counterfactual embedding target. |

### `data_generator`

```python
data_generator(X: numpy.ndarray, encoder_preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int, shuffle: bool = True, kwargs)
```

Constructs a `tensorflow` data generator.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling the `preprocessor` function. |
| `encoder_preprocessor` | `Callable` |  | Preprocessor function. This function correspond to the preprocessing steps applied to the encoder/auto-encoder model. |
| `predictor` | `Callable` |  | Prediction function. The classifier function should expect the input in the original format and preprocess it internally in the `predictor` if necessary. |
| `conditional_func` | `Callable` |  | Conditional function generator. Given an preprocessed input array, the functions generates a conditional array. |
| `batch_size` | `int` |  | Dimension of the batch used during training. The same batch size is used to infer the classification labels of the input dataset. |
| `shuffle` | `bool` | `True` | Whether to shuffle the dataset each epoch. ``True`` by default. |
| `**kwargs` |  |  | Other arguments. Not used. |

### `decode`

```python
decode(Z: Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray], decoder: keras.src.models.model.Model, kwargs)
```

Decodes an embedding tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  | Embedding tensor to be decoded. |
| `decoder` | `keras.src.models.model.Model` |  | Pretrained decoder network. |
| `**kwargs` |  |  | Other arguments. Not used. |

### `encode`

```python
encode(X: Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray], encoder: keras.src.models.model.Model, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

Encodes the input tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[tensorflow.python.framework.tensor.Tensor, numpy.ndarray]` |  | Input to be encoded. |
| `encoder` | `keras.src.models.model.Model` |  | Pretrained encoder network. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `generate_cf`

```python
generate_cf(Z: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], Y_m: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], Y_t: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], C: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None], actor: keras.src.models.model.Model, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

Generates counterfactual embedding.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Input embedding tensor. |
| `Y_m` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Input classification label. |
| `Y_t` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Target counterfactual classification label. |
| `C` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]` |  | Conditional tensor. |
| `actor` | `keras.src.models.model.Model` |  | Actor network. The model generates the counterfactual embedding. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `get_actor`

```python
get_actor(hidden_dim: int, output_dim: int) -> keras.src.layers.layer.Layer
```

Constructs the actor network.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Actor's hidden dimension |
| `output_dim` | `int` |  | Actor's output dimension. |

**Returns**
- Type: `keras.src.layers.layer.Layer`

### `get_critic`

```python
get_critic(hidden_dim: int) -> keras.src.layers.layer.Layer
```

Constructs the critic network.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Critic's hidden dimension. |

**Returns**
- Type: `keras.src.layers.layer.Layer`

### `get_optimizer`

```python
get_optimizer(model: Optional[keras.src.layers.layer.Layer] = None, lr: float = 0.001) -> keras.src.optimizers.optimizer.Optimizer
```

Constructs default `Adam` optimizer.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Optional[keras.src.layers.layer.Layer]` | `None` | Model to get the optimizer for. Not required for `tensorflow` backend. |
| `lr` | `float` | `0.001` | Learning rate. |

**Returns**
- Type: `keras.src.optimizers.optimizer.Optimizer`

### `initialize_actor_critic`

```python
initialize_actor_critic(actor, critic, Z, Z_cf_tilde, Y_m, Y_t, C, kwargs)
```

Initialize actor and critic layers by passing a dummy zero tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `actor` |  |  | Actor model. |
| `critic` |  |  | Critic model. |
| `Z` |  |  | Input embedding. |
| `Z_cf_tilde` |  |  | Noised counterfactual embedding. |
| `Y_m` |  |  | Input classification label. |
| `Y_t` |  |  | Target counterfactual classification label. |
| `C` |  |  | Conditional tensor. |
| `**kwargs` |  |  | Other arguments. Not used. |

### `initialize_optimizer`

```python
initialize_optimizer(optimizer: keras.src.optimizers.optimizer.Optimizer, model: keras.src.models.model.Model) -> None
```

Initializes an optimizer given a model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `keras.src.optimizers.optimizer.Optimizer` |  | Optimizer to be initialized. |
| `model` | `keras.src.models.model.Model` |  | Model to be optimized |

**Returns**
- Type: `None`

### `initialize_optimizers`

```python
initialize_optimizers(optimizer_actor, optimizer_critic, actor, critic, kwargs) -> None
```

Initializes the actor and critic optimizers.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer_actor` |  |  | Actor optimizer to be initialized. |
| `optimizer_critic` |  |  | Critic optimizer to be initialized. |
| `actor` |  |  | Actor model to be optimized. |
| `critic` |  |  | Critic model to be optimized. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `None`

### `load_model`

```python
load_model(path: Union[str, os.PathLike]) -> keras.src.models.model.Model
```

Loads a model and its optimizer.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to the loading location. |

**Returns**
- Type: `keras.src.models.model.Model`

### `save_model`

```python
save_model(path: Union[str, os.PathLike], model: keras.src.layers.layer.Layer) -> None
```

Saves a model and its optimizer.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to the saving location. |
| `model` | `keras.src.layers.layer.Layer` |  | Model to be saved. |

**Returns**
- Type: `None`

### `set_seed`

```python
set_seed(seed: int = 13)
```

Sets a seed to ensure reproducibility. Does NOT ensure reproducibility.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `seed` | `int` | `13` | seed to be set |

### `sparsity_loss`

```python
sparsity_loss(X_hat_cf: tensorflow.python.framework.tensor.Tensor, X: tensorflow.python.framework.tensor.Tensor) -> Dict[str, tensorflow.python.framework.tensor.Tensor]
```

Default L1 sparsity loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cf` | `tensorflow.python.framework.tensor.Tensor` |  | Auto-encoder counterfactual reconstruction. |
| `X` | `tensorflow.python.framework.tensor.Tensor` |  | Input instance. |

**Returns**
- Type: `Dict[str, tensorflow.python.framework.tensor.Tensor]`

### `to_numpy`

```python
to_numpy(X: Union[List[Any], numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]) -> Union[List[Any], numpy.ndarray, None]
```

Converts given tensor to `numpy` array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[List[Any], numpy.ndarray, tensorflow.python.framework.tensor.Tensor, None]` |  | Input tensor to be converted to `numpy` array. |

**Returns**
- Type: `Union[List[Any], numpy.ndarray, None]`

### `to_tensor`

```python
to_tensor(X: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], kwargs) -> Optional[tensorflow.python.framework.tensor.Tensor]
```

Converts tensor to `tf.Tensor`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Input array/tensor to be converted. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `Optional[tensorflow.python.framework.tensor.Tensor]`
