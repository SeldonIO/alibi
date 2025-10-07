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
DEFAULT_DATA_SIM: dict = {'scores': None, 'ordered_indices': None, 'most_similar': None, 'least_simila...
```

### `DEFAULT_META_SIM`
```python
DEFAULT_META_SIM: dict = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```

## `GradientSimilarity`

_Inherits from:_ `BaseSimilarityExplainer`, `Explainer`, `ABC`, `Base`

Base class for similarity explainers.

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

### `explain`

```python
explain(X: Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]], Y: Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]] = None) -> Explanation
```

Explain the predictor's predictions for a given input.

Computes the similarity score between the inputs and the training set. Returns an explainer object
containing the scores, the indices of the training set instances sorted by descending similarity and the
most similar and least similar instances of the data set for the input. Note that the input may be a single
instance or a batch of instances.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]` |  |  |
| `Y` | `Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]` | `None` |  |

**Returns**
- Type: `Explanation`

### `fit`

```python
fit(X_train: Union[numpy.ndarray, List[typing.Any]], Y_train: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

Fit the explainer.

The `GradientSimilarity` explainer requires the model gradients over the training data. In the explain method
it compares them to the model gradients for the test instance(s). If ``precompute_grads=True`` on
initialization then the gradients are precomputed here and stored. This will speed up the explain method call
but storing the gradients may not be feasible for large models.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `Union[numpy.ndarray, List[typing.Any]]` |  |  |
| `Y_train` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

## `Task`

_Inherits from:_ `str`, `Enum`

Enum of supported tasks.

### Constructor

```python
Task(self, /, *args, **kwargs)
```
