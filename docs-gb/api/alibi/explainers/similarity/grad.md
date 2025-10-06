# `alibi.explainers.similarity.grad`

Gradient-based explainer.
This module implements the gradient-based explainers grad-dot and grad-cos.

## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `DEFAULT_DATA_SIM`
```python
DEFAULT_DATA_SIM = {'scores': None, 'ordered_indices': None, 'most_similar': None, 'least_simila...
```

### `DEFAULT_META_SIM`
```python
DEFAULT_META_SIM = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```

## Classes
### `GradientSimilarity` (_inherits from `BaseSimilarityExplainer`, `Explainer`, `ABC`, `Base`)

Base class for similarity explainers.

#### Constructor

```python
GradientSimilarity(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]', loss_fn: 'Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],\n                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]', sim_fn: typing_extensions.Literal['grad_dot', 'grad_cos', 'grad_asym_dot'] = 'grad_dot', task: typing_extensions.Literal['classification', 'regression'] = 'classification', precompute_grads: bool = False, backend: typing_extensions.Literal['tensorflow', 'pytorch'] = 'tensorflow', device: 'Union[int, str, torch.device, None]' = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Model to explain. |
| `loss_fn` | `Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]` |  | Loss function used. The gradient of the loss function is used to compute the similarity between the test instances and the training set. |
| `sim_fn` | `Literal[grad_dot, grad_cos, grad_asym_dot]` | `'grad_dot'` | Similarity function to use. The ``'grad_dot'`` similarity function computes the dot product of the gradients, see :py:func:`alibi.explainers.similarity.metrics.dot`. The ``'grad_cos'`` similarity function computes the cosine similarity between the gradients, see :py:func:`alibi.explainers.similarity.metrics.cos`. The ``'grad_asym_dot'`` similarity function is similar to ``'grad_dot'`` but is asymmetric, see :py:func:`alibi.explainers.similarity.metrics.asym_dot`. |
| `task` | `Literal[classification, regression]` | `'classification'` | Type of task performed by the model. If the task is ``'classification'``, the target value passed to the explain method of the test instance can be specified either directly or left  as ``None``, if left ``None`` we use the model's maximum prediction. If the task is ``'regression'``, the target value of the test instance must be specified directly. |
| `precompute_grads` | `bool` | `False` | Whether to precompute the gradients. If ``False``, gradients are computed on the fly otherwise we precompute them which can be faster when it comes to computing explanations. Note this option may be memory intensive if the model is large. |
| `backend` | `Literal[tensorflow, pytorch]` | `'tensorflow'` | Backend to use. |
| `device` | `Union[int, str, torch.device, None]` | `None` | Device to use. If ``None``, the default device for the backend is used. If using `pytorch` backend see `pytorch device docs <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_ for correct options. Note that in the `pytorch` backend case this parameter can be a ``torch.device``. If using `tensorflow` backend see `tensorflow docs <https://www.tensorflow.org/api_docs/python/tf/device>`_ for correct options. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

#### Methods

##### `explain`

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
| `X` | `Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]` |  | `X` can be a `numpy` array, `tensorflow` tensor, `pytorch` tensor of the same shape as the training data or a list of objects, with or without a leading batch dimension. If the batch dimension is missing it's added. |
| `Y` | `Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]` | `None` | `Y` can be a `numpy` array, `tensorflow` tensor or a `pytorch` tensor. In the case of a regression task, the `Y` argument must be present. If the task is classification then `Y` defaults to the model prediction. |

**Returns**
- Type: `Explanation`

##### `fit`

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
| `X_train` | `Union[numpy.ndarray, List[typing.Any]]` |  | Training data. |
| `Y_train` | `numpy.ndarray` |  | Training labels. |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

### `Task` (_inherits from `str`, `Enum`)

Enum of supported tasks.

#### Constructor

```python
Task(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |
