# `alibi.explainers.similarity.base`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

## `BaseSimilarityExplainer`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for similarity explainers.

### Constructor

```python
BaseSimilarityExplainer(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]', loss_fn: 'Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],\n                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]', sim_fn: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray], precompute_grads: bool = False, backend: alibi.utils.frameworks.Framework = <Framework.TENSORFLOW: 'tensorflow'>, device: 'Union[int, str, torch.device, None]' = None, meta: Optional[dict] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Model to be explained. |
| `loss_fn` | `Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]` |  | Loss function. |
| `sim_fn` | `Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray]` |  | Similarity function. Takes two inputs and returns a similarity value. |
| `precompute_grads` | `bool` | `False` | Whether to precompute and store the gradients when fitting. |
| `backend` | `alibi.utils.frameworks.Framework` | `<Framework.TENSORFLOW: 'tensorflow'>` | Deep learning backend. |
| `device` | `Union[int, str, torch.device, None]` | `None` | Device to be used. Will default to the same device the backend defaults to. |
| `meta` | `Optional[dict]` | `None` | Metadata specific to explainers that inherit from this class. Should be initialized in the child class and passed in here. Is used in the `__init__` of the base Explainer class. |
| `verbose` | `bool` | `False` |  |

### Methods

#### `fit`

```python
fit(X_train: Union[numpy.ndarray, List[typing.Any]], Y_train: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_train` | `Union[numpy.ndarray, List[typing.Any]]` |  | Training data. |
| `Y_train` | `numpy.ndarray` |  | Training labels. |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[tensorflow.keras.Model, torch.nn.Module]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | The new predictor to use. |

**Returns**
- Type: `None`
