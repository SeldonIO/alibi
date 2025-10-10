# `alibi.explainers.backends.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base`, that are common for both Tensorflow and Pytorch backends.

## `CounterfactualRLDataset`

_Inherits from:_ `ABC`

### Methods

#### `predict_batches`

```python
predict_batches(X: numpy.ndarray, predictor: Callable, batch_size: int) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Input to be classified. |
| `predictor` | `Callable` |  | Prediction function. |
| `batch_size` | `int` |  | Maximum batch size to be used during each inference step. |

**Returns**
- Type: `numpy.ndarray`

## Functions
### `generate_empty_condition`

```python
generate_empty_condition(X: typing.Any) -> None
```

Empty conditioning.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  | Input instance. |

**Returns**
- Type: `None`

### `get_classification_reward`

```python
get_classification_reward(Y_pred: numpy.ndarray, Y_true: numpy.ndarray)
```

Computes classification reward per instance given the prediction output and the true label. The classification
reward is a sparse/binary reward: 1 if the most likely classes from the prediction output and the label match,
0 otherwise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Y_pred` | `numpy.ndarray` |  | Prediction output as a distribution over the possible classes. |
| `Y_true` | `numpy.ndarray` |  | True label as a distribution over the possible classes. |

### `get_hard_distribution`

```python
get_hard_distribution(Y: numpy.ndarray, num_classes: Optional[int] = None) -> numpy.ndarray
```

Constructs the hard label distribution (one-hot encoding).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Y` | `numpy.ndarray` |  | Prediction array. Can be soft or hard label distribution, or a label. |
| `num_classes` | `Optional[int]` | `None` | Number of classes to be considered. |

**Returns**
- Type: `numpy.ndarray`

### `identity_function`

```python
identity_function(X: typing.Any) -> typing.Any
```

Identity function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  | Input instance. |

**Returns**
- Type: `typing.Any`
