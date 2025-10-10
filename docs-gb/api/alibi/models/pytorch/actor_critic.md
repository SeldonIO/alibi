# `alibi.models.pytorch.actor_critic`

This module contains the Pytorch implementation of actor-critic networks used in the Counterfactual with Reinforcement
Learning for both data modalities. The models' architectures follow the standard actor-critic design and can have
broader use-cases.

## `Actor`

_Inherits from:_ `Module`

Actor network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

### Constructor

```python
Actor(self, hidden_dim: int, output_dim: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |
| `output_dim` | `int` |  | Output dimension |

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

## `Critic`

_Inherits from:_ `Module`

Critic network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.

### Constructor

```python
Critic(self, hidden_dim: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_dim` | `int` |  | Hidden dimension. |

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
