# `alibi.models.tensorflow.cfrl_models`

This module contains the Tensorflow implementation of models used for the Counterfactual with Reinforcement Learning
experiments for both data modalities (image and tabular).

## `ADULTDecoder`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

ADULT decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
of a fully connected layer with ReLU nonlinearity, and a multiheaded layer, one for each categorical feature and
a single head for the rest of numerical features. The hidden dimension used in the paper is 128.

### Constructor

```python
ADULTDecoder(self, hidden_dim: int, output_dims: List[int], **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |
| `output_dims` | `List[int]` |  |  |
| `output_dim` |  |  | List of output dimensions. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> List[tensorflow.python.framework.tensor.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `List[tensorflow.python.framework.tensor.Tensor]`

## `ADULTEncoder`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

ADULT encoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
two fully connected layers with ReLU and tanh nonlinearities. The tanh nonlinearity clips the embedding in [-1, 1]
as required in the DDPG algorithm (e.g., [act_low, act_high]). The layers' dimensions used in the paper are
128 and 15, although those can vary as they were selected to generalize across many datasets.

### Constructor

```python
ADULTEncoder(self, hidden_dim: int, latent_dim: int, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |
| `latent_dim` | `int` |  | Latent dimension. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `**kwargs` |  |  | Other arguments. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## `MNISTClassifier`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

MNIST classifier used in the experiments for Counterfactual with Reinforcement Learning. The model consists of two
convolutional layers having 64 and 32 channels and a kernel size of 2 with ReLU nonlinearities, followed by
maxpooling of size 2 and dropout of 0.3. The convolutional block is followed by a fully connected layer of 256 with
ReLU nonlinearity, and finally a fully connected layer is used to predict the class logits (10 in MNIST case).

### Constructor

```python
MNISTClassifier(self, output_dim: int = 10, **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `output_dim` | `int` | `10` | Output dimension |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, training: bool = True, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `training` | `bool` | `True` | Training flag. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## `MNISTDecoder`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
except the last one, has ReLU nonlinearities and is followed by an up-sampling layer of size 2. The final layers
uses a sigmoid activation to clip the output values in [0, 1].

### Constructor

```python
MNISTDecoder(self, **kwargs) -> None
```
### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## `MNISTEncoder`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
this can vary.

### Constructor

```python
MNISTEncoder(self, latent_dim: int, **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `latent_dim` | `int` |  | Latent dimension. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `**kwargs` |  |  | Other arguments. Not used. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
