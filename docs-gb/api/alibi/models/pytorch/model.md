# `alibi.models.pytorch.model`

This module tries to provided a class wrapper to mimic the TensorFlow API of `tensorflow.keras.Model`. It
is intended to simplify the training of a model through methods like compile, fit and evaluate which allow the user
to define custom loss functions, optimizers, evaluation metrics, train a model and evaluate it. Currently it is
used internally to test the functionalities for the Pytorch backend. To be discussed if the module will be exposed
to the user in future versions.

## `Model`

_Inherits from:_ `Module`

### Constructor

```python
Model(self, **kwargs)
```
### Methods

#### `compile`

```python
compile(optimizer: torch.optim.optimizer.Optimizer, loss: Union[Callable, List[Callable]], loss_weights: Optional[List[float]] = None, metrics: Optional[List[alibi.models.pytorch.metrics.Metric]] = None)
```

Compiles a model by setting the optimizer and the loss functions, loss weights and metrics to monitor

the training of the model.

Parameters
----------
optimizer
    Optimizer to be used.
loss
    Loss function to be used. Can be a list of the loss function which will be weighted and summed up to
    compute the total loss.
loss_weights
    Weights corresponding to each loss function. Only used if the `loss` argument is a  list.
metrics
    Metrics used to monitor the training process.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `torch.optim.optimizer.Optimizer` |  |  |
| `loss` | `Union[Callable, List[Callable]]` |  |  |
| `loss_weights` | `Optional[List[float]]` | `None` |  |
| `metrics` | `Optional[List[alibi.models.pytorch.metrics.Metric]]` | `None` |  |

#### `compute_loss`

```python
compute_loss(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]
```

Computes the loss given the prediction labels and the true labels.

Parameters
---------
y_pred
    Prediction labels.
y_true
    True labels.

Returns
-------
A tuple consisting of the total loss computed as a weighted sum of individual losses and a dictionary         of individual losses used of logging.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Tuple[torch.Tensor, Dict[str, float]]`

#### `compute_metrics`

```python
compute_metrics(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

Computes the metrics given the prediction labels and the true labels.

Parameters
----------
y_pred
    Prediction labels.
y_true
    True labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Dict[str, float]`

#### `evaluate`

```python
evaluate(testloader: torch.utils.data.dataloader.DataLoader) -> Dict[str, float]
```

Evaluation function. The function reports the evaluation metrics used for monitoring the training loop.

Parameters
----------
testloader
    Test dataloader.

Returns
-------
Evaluation metrics.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `testloader` | `torch.utils.data.dataloader.DataLoader` |  |  |

**Returns**
- Type: `Dict[str, float]`

#### `fit`

```python
fit(trainloader: torch.utils.data.dataloader.DataLoader, epochs: int) -> Dict[str, float]
```

Fit method. Equivalent of a training loop.

Parameters
----------
trainloader
    Training data loader.
epochs
    Number of epochs to train the model.

Returns
-------
Final epoch monitoring metrics.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `trainloader` | `torch.utils.data.dataloader.DataLoader` |  |  |
| `epochs` | `int` |  |  |

**Returns**
- Type: `Dict[str, float]`

#### `load_weights`

```python
load_weights(path: str) -> None
```

Loads the weight of the current model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `str` |  |  |

**Returns**
- Type: `None`

#### `save_weights`

```python
save_weights(path: str) -> None
```

Save the weight of the current model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `str` |  |  |

**Returns**
- Type: `None`

#### `test_step`

```python
test_step(x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]])
```

Performs a test step.

Parameters
----------
x
    Input tensor.
y
    Label tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

#### `train_step`

```python
train_step(x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

Performs a train step.

Parameters
----------
x
    Input tensor.
y
    Label tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |

**Returns**
- Type: `Dict[str, float]`

#### `validate_prediction_labels`

```python
validate_prediction_labels(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]])
```

Validates the loss functions, loss weights, training labels and prediction labels.

Parameters
---------
y_pred
    Prediction labels.
y_true
    True labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  |  |
