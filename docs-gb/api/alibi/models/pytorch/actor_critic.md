# `alibi.models.pytorch.actor_critic`

This module contains the Pytorch implementation of actor-critic networks used in the Counterfactual with Reinforcement
Learning for both data modalities. The models' architectures follow the standard actor-critic design and can have
broader use-cases.

## Classes
### `Actor` (_inherits from `Module`)

Actor network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.

The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

#### Constructor

```python
Actor(self, hidden_dim: int, output_dim: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |
| `output_dim` | `int` |  |  |

#### Methods

##### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Forward pass

Parameters
----------
x
    Input tensor.

Returns
-------
Continuous action.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |

**Returns**
- Type: `torch.Tensor`

### `Critic` (_inherits from `Module`)

Critic network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.

The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

#### Constructor

```python
Critic(self, hidden_dim: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  |  |

#### Methods

##### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Forward pass.

Parameters
----------
x
    Input tensor.

Returns
-------
Critic value.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |

**Returns**
- Type: `torch.Tensor`
