# `alibi.models.tensorflow.actor_critic`

This module contains the Tensorflow implementation of actor-critic networks used in the Counterfactual with
Reinforcement Learning for both data modalities. The models' architectures follow the standard actor-critic design and
can have broader use-cases.

## `Actor`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Actor network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

### Constructor

```python
Actor(self, hidden_dim: int, output_dim: int, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension |
| `output_dim` | `int` |  | Output dimension |

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

#### `from_config`

```python
from_config(config)
```

Creates the model from its configuration.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` |  |  |  |

#### `get_config`

```python
get_config()
```

Returns the configuration of the model for serialization.

## `Critic`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Critic network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

### Constructor

```python
Critic(self, hidden_dim: int, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

#### `from_config`

```python
from_config(config)
```

Creates the model from its configuration.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` |  |  |  |

#### `get_config`

```python
get_config()
```

Returns the configuration of the model for serialization.
