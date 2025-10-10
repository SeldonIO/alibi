# `alibi.models.pytorch.cfrl_models`

This module contains the Pytorch implementation of models used for the Counterfactual with Reinforcement Learning
experiments for both data modalities (image and tabular).

## `ADULTDecoder`

_Inherits from:_ `Module`

ADULT decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
of a fully connected layer with ReLU nonlinearity, and a multiheaded layer, one for each categorical feature and
a single head for the rest of numerical features. The hidden dimension used in the paper is 128.

### Constructor

```python
ADULTDecoder(self, hidden_dim: int, output_dims: List[int])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |
| `output_dims` | `List[int]` |  | List of output dimensions. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> List[torch.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `List[torch.Tensor]`

## `ADULTEncoder`

_Inherits from:_ `Module`

ADULT encoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
two fully connected layers with ReLU and tanh nonlinearities. The tanh nonlinearity clips the embedding in [-1, 1]
as required in the DDPG algorithm (e.g., [act_low, act_high]). The layers' dimensions used in the paper are
128 and 15, although those can vary as they were selected to generalize across many datasets.

### Constructor

```python
ADULTEncoder(self, hidden_dim: int, latent_dim: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |
| `latent_dim` | `int` |  | Latent dimension. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `torch.Tensor`

## `MNISTClassifier`

_Inherits from:_ `Model`, `Module`

MNIST classifier used in the experiments for Counterfactual with Reinforcement Learning. The model consists of two
convolutional layers having 64 and 32 channels and a kernel size of 2 with ReLU nonlinearities, followed by
maxpooling of size 2 and dropout of 0.3. The convolutional block is followed by a fully connected layer of 256 with
ReLU nonlinearity, and finally a fully connected layer is used to predict the class logits (10 in MNIST case).

### Constructor

```python
MNISTClassifier(self, output_dim: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `output_dim` | `int` |  | Output dimension. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `torch.Tensor`

## `MNISTDecoder`

_Inherits from:_ `Module`

MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
except the last one, has ReLU nonlinearities and is followed by an upsampling layer of size 2. The final layers
uses a sigmoid activation to clip the output values in [0, 1].

### Constructor

```python
MNISTDecoder(self, latent_dim: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `latent_dim` | `int` |  | Latent dimension. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `torch.Tensor`

## `MNISTEncoder`

_Inherits from:_ `Module`

MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
this can vary.

### Constructor

```python
MNISTEncoder(self, latent_dim: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `latent_dim` | `int` |  | Latent dimension. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `torch.Tensor`
