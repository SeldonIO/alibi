# `alibi.models.pytorch.metrics`

This module contains a loss wrapper and a definition of various monitoring metrics used during training. The model
to be trained inherits form :py:class:`alibi.explainers.models.pytorch.model.Model` and represents a simplified
version of the `tensorflow.keras` API for training and monitoring the model. Currently it is used internally to test
the functionalities for the Pytorch backend. To be discussed if the module will be exposed to the user in future
versions.

## `AccuracyMetric`

_Inherits from:_ `Metric`, `ABC`

Accuracy monitoring metric.

### Constructor

```python
AccuracyMetric(self, name: str = 'accuracy')
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `name` | `str` | `'accuracy'` | Name of the metric. |
| `reduction` |  |  | Metric's reduction type. Possible values `mean`|`sum`. By default `mean`. |

### Methods

#### `compute_metric`

```python
compute_metric(y_pred: Union[torch.Tensor, numpy.ndarray], y_true: Union[torch.Tensor, numpy.ndarray]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, numpy.ndarray]` |  | Predicted label. |
| `y_true` | `Union[torch.Tensor, numpy.ndarray]` |  | True label. |

**Returns**
- Type: `None`

## `LossContainer`

Loss wrapped to monitor the average loss throughout training.

### Constructor

```python
LossContainer(self, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], name: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `loss` | `Callable[[.[<class 'torch.Tensor'>, <class 'torch.Tensor'>]], torch.Tensor]` |  | Loss function. |
| `name` | `str` |  | Name of the loss function |

### Methods

#### `reset`

```python
reset()
```

Resets the loss.

#### `result`

```python
result() -> Dict[str, float]
```

Computes the average loss obtain by dividing the cumulated loss by the number of steps

**Returns**
- Type: `Dict[str, float]`

## `Metric`

_Inherits from:_ `ABC`

Monitoring metric object. Supports two types of reduction: mean and sum.

### Constructor

```python
Metric(self, reduction: alibi.models.pytorch.metrics.Reduction = <Reduction.MEAN: 'mean'>, name: str = 'unknown')
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `reduction` | `alibi.models.pytorch.metrics.Reduction` | `<Reduction.MEAN: 'mean'>` | Metric's reduction type. Possible values `mean`|`sum`. By default `mean`. |
| `name` | `str` | `'unknown'` | Name of the metric. |

### Methods

#### `compute_metric`

```python
compute_metric(y_pred: Union[torch.Tensor, numpy.ndarray], y_true: Union[torch.Tensor, numpy.ndarray])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, numpy.ndarray]` |  |  |
| `y_true` | `Union[torch.Tensor, numpy.ndarray]` |  |  |

#### `reset`

```python
reset()
```

Resets the monitoring metric.

#### `result`

```python
result() -> Dict[str, float]
```

Computes the result according to the reduction procedure.

**Returns**
- Type: `Dict[str, float]`

#### `update_state`

```python
update_state(values: numpy.ndarray)
```

Update the state of the metric by summing up the metric values and updating the counts by adding
the number of instances for which the metric was computed (first dimension).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |

## `Reduction`

_Inherits from:_ `Enum`

Reduction operation supported by the monitoring metrics.
