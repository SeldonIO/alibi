# `alibi.models.pytorch.model`

This module tries to provided a class wrapper to mimic the TensorFlow API of `tensorflow.keras.Model`. It
is intended to simplify the training of a model through methods like compile, fit and evaluate which allow the user
to define custom loss functions, optimizers, evaluation metrics, train a model and evaluate it. Currently it is
used internally to test the functionalities for the Pytorch backend. To be discussed if the module will be exposed
to the user in future versions.

## `Model`

_Inherits from:_ `Module`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

### Constructor

```python
Model(self, **kwargs)
```
### Methods

### `compile`

```python
compile(optimizer: torch.optim.optimizer.Optimizer, loss: Union[Callable, List[Callable]], loss_weights: Optional[List[float]] = None, metrics: Optional[List[alibi.models.pytorch.metrics.Metric]] = None)
```

Compiles a model by setting the optimizer and the loss functions, loss weights and metrics to monitor

the training of the model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `torch.optim.optimizer.Optimizer` |  |  |
| `loss` | `Union[Callable, List[Callable]]` |  |  |
| `loss_weights` | `Optional[List[float]]` | `None` |  |
| `metrics` | `Optional[List[alibi.models.pytorch.metrics.Metric]]` | `None` |  |

### `compute_loss`

```python
compute_loss(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]
```

Computes the loss given the prediction labels and the true labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Tuple[torch.Tensor, Dict[str, float]]`

### `compute_metrics`

```python
compute_metrics(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

Computes the metrics given the prediction labels and the true labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Dict[str, float]`

### `evaluate`

```python
evaluate(testloader: torch.utils.data.dataloader.DataLoader) -> Dict[str, float]
```

Evaluation function. The function reports the evaluation metrics used for monitoring the training loop.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `testloader` | `torch.utils.data.dataloader.DataLoader` |  |  |

**Returns**
- Type: `Dict[str, float]`

### `fit`

```python
fit(trainloader: torch.utils.data.dataloader.DataLoader, epochs: int) -> Dict[str, float]
```

Fit method. Equivalent of a training loop.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `trainloader` | `torch.utils.data.dataloader.DataLoader` |  |  |
| `epochs` | `int` |  |  |

**Returns**
- Type: `Dict[str, float]`

### `load_weights`

```python
load_weights(path: str) -> None
```

Loads the weight of the current model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `str` |  |  |

**Returns**
- Type: `None`

### `save_weights`

```python
save_weights(path: str) -> None
```

Save the weight of the current model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `str` |  |  |

**Returns**
- Type: `None`

### `test_step`

```python
test_step(x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]])
```

Performs a test step.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

### `train_step`

```python
train_step(x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

Performs a train step.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Dict[str, float]`

### `validate_prediction_labels`

```python
validate_prediction_labels(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]])
```

Validates the loss functions, loss weights, training labels and prediction labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
