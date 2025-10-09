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

Predict the classification labels of the input dataset. This is performed in batches.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `predictor` | `Callable` |  |  |
| `batch_size` | `int` |  |  |

**Returns**
- Type: `numpy.ndarray`
