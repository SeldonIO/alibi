# `alibi.explainers.backends.pytorch.cfrl_base`

This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base` for the Pytorch backend.

## `PtCounterfactualRLDataset`

_Inherits from:_ `CounterfactualRLDataset`, `ABC`, `Dataset`, `Generic`

Pytorch backend datasets.

### Constructor

```python
PtCounterfactualRLDataset(self, X: numpy.ndarray, preprocessor: Callable, predictor: Callable, conditional_func: Callable, batch_size: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `preprocessor` | `Callable` |  |  |
| `predictor` | `Callable` |  |  |
| `conditional_func` | `Callable` |  |  |
| `batch_size` | `int` |  |  |
