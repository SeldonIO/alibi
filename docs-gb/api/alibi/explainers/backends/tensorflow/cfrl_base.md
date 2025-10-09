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
