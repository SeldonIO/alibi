# `alibi.explainers.similarity.grad`

Gradient-based explainer.
This module implements the gradient-based explainers grad-dot and grad-cos.

## `GradientSimilarity`

_Inherits from:_ `BaseSimilarityExplainer`, `Explainer`, `ABC`, `Base`

### Constructor

```python
GradientSimilarity(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]', loss_fn: 'Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],\n                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]', sim_fn: typing_extensions.Literal['grad_dot', 'grad_cos', 'grad_asym_dot'] = 'grad_dot', task: typing_extensions.Literal['classification', 'regression'] = 'classification', precompute_grads: bool = False, backend: typing_extensions.Literal['tensorflow', 'pytorch'] = 'tensorflow', device: 'Union[int, str, torch.device, None]' = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `loss_fn` | `Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]` |  |  |
| `sim_fn` | `Literal[grad_dot, grad_cos, grad_asym_dot]` | `'grad_dot'` |  |
| `task` | `Literal[classification, regression]` | `'classification'` |  |
| `precompute_grads` | `bool` | `False` |  |
| `backend` | `Literal[tensorflow, pytorch]` | `'tensorflow'` |  |
| `device` | `Union[int, str, torch.device, None]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `explain`

```python
explain(X: Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]], Y: Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]] = None) -> Explanation
```

Explain the predictor's predictions for a given input.

Computes the similarity score between the inputs and the training set. Returns an explainer object
containing the scores, the indices of the training set instances sorted by descending similarity and the
most similar and least similar instances of the data set for the input. Note that the input may be a single
instance or a batch of instances.

Parameters
----------
X
    `X` can be a `numpy` array, `tensorflow` tensor, `pytorch` tensor of the same shape as the training data
    or a list of objects, with or without a leading batch dimension. If the batch dimension is missing it's
    added.
Y
    `Y` can be a `numpy` array, `tensorflow` tensor or a `pytorch` tensor. In the case of a regression task, the
    `Y` argument must be present. If the task is classification then `Y` defaults to the model prediction.

Returns
-------
`Explanation` object containing the ordered similarity scores for the test instance(s) with additional         metadata as attributes. Contains the following data-related attributes
    - `scores`: ``np.ndarray`` - similarity scores for each pair of instances in the training and test set             sorted in descending order.
    - `ordered_indices`: ``np.ndarray`` - indices of the paired training and test set instances sorted by the             similarity score in descending order.
    - `most_similar`: ``np.ndarray`` - 5 most similar instances in the training set for each test instance             The first element is the most similar instance.
    -  `least_similar`: ``np.ndarray`` - 5 least similar instances in the training set for each test instance.             The first element is the least similar instance.

Raises
------
ValueError
    If `Y` is ``None`` and the `task` is ``'regression'``.
ValueError
    If the shape of `X` or `Y` does not match the shape of the training or target data.
ValueError
    If the fit method has not been called prior to calling this method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]` |  |  |
| `Y` | `Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]` | `None` |  |

**Returns**
- Type: `Explanation`

#### `fit`

```python
fit(X_train: Union[numpy.ndarray, List[typing.Any]], Y_train: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

Fit the explainer.

The `GradientSimilarity` explainer requires the model gradients over the training data. In the explain method
it compares them to the model gradients for the test instance(s). If ``precompute_grads=True`` on
initialization then the gradients are precomputed here and stored. This will speed up the explain method call
but storing the gradients may not be feasible for large models.

Parameters
----------
X_train
    Training data.
Y_train
    Training labels.

Returns
-------
self
    Returns self.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `Union[numpy.ndarray, List[typing.Any]]` |  |  |
| `Y_train` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

## `Task`

_Inherits from:_ `str`, `Enum`

Enum of supported tasks.
