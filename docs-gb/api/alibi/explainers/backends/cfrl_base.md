# `alibi.explainers.backends.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base`, that are common for both Tensorflow and Pytorch backends.

## `CounterfactualRLDataset`

_Inherits from:_ `ABC`

Helper class that provides a standard way to create an ABC using

inheritance.

### Constructor

```python
CounterfactualRLDataset(self, /, *args, **kwargs)
```
### Methods

### `predict_batches`

```python
predict_batches(X: numpy.ndarray, predictor: Callable, batch_size: int) -> numpy.ndarray
```

Predict the classification labels of the input dataset. This is performed in batches.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `predictor` | `Callable` |  |  |
| `batch_size` | `int` |  |  |

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
| `X` | `typing.Any` |  |  |

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
| `Y_pred` | `numpy.ndarray` |  |  |
| `Y_true` | `numpy.ndarray` |  |  |

### `get_hard_distribution`

```python
get_hard_distribution(Y: numpy.ndarray, num_classes: Optional[int] = None) -> numpy.ndarray
```

Constructs the hard label distribution (one-hot encoding).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Y` | `numpy.ndarray` |  |  |
| `num_classes` | `Optional[int]` | `None` |  |

**Returns**
- Type: `numpy.ndarray`

### `identity_function`

```python
identity_function(X: typing.Any) -> typing.Any
```

Identity function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  |  |

**Returns**
- Type: `typing.Any`
