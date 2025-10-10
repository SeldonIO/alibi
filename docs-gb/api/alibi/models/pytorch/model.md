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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `torch.optim.optimizer.Optimizer` |  | Optimizer to be used. |
| `loss` | `Union[Callable, List[Callable]]` |  | Loss function to be used. Can be a list of the loss function which will be weighted and summed up to compute the total loss. |
| `loss_weights` | `Optional[List[float]]` | `None` | Weights corresponding to each loss function. Only used if the `loss` argument is a  list. |
| `metrics` | `Optional[List[alibi.models.pytorch.metrics.Metric]]` | `None` | Metrics used to monitor the training process. |

#### `compute_loss`

```python
compute_loss(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  | Prediction labels. |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  | True labels. |

**Returns**
- Type: `Tuple[torch.Tensor, Dict[str, float]]`

#### `compute_metrics`

```python
compute_metrics(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  | Prediction labels. |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  | True labels. |

**Returns**
- Type: `Dict[str, float]`

#### `evaluate`

```python
evaluate(testloader: torch.utils.data.dataloader.DataLoader) -> Dict[str, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `testloader` | `torch.utils.data.dataloader.DataLoader` |  | Test dataloader. |

**Returns**
- Type: `Dict[str, float]`

#### `fit`

```python
fit(trainloader: torch.utils.data.dataloader.DataLoader, epochs: int) -> Dict[str, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `trainloader` | `torch.utils.data.dataloader.DataLoader` |  | Training data loader. |
| `epochs` | `int` |  | Number of epochs to train the model. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  | Label tensor. |

#### `train_step`

```python
train_step(x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |
| `y` | `Union[torch.Tensor, List[torch.Tensor]]` |  | Label tensor. |

**Returns**
- Type: `Dict[str, float]`

#### `validate_prediction_labels`

```python
validate_prediction_labels(y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Union[torch.Tensor, List[torch.Tensor]])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_pred` | `Union[torch.Tensor, List[torch.Tensor]]` |  | Prediction labels. |
| `y_true` | `Union[torch.Tensor, List[torch.Tensor]]` |  | True labels. |
