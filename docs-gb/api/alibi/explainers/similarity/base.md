# `alibi.explainers.similarity.base`
## `BaseSimilarityExplainer`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for similarity explainers.

### Constructor

```python
BaseSimilarityExplainer(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]', loss_fn: 'Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],\n                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]', sim_fn: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray], precompute_grads: bool = False, backend: alibi.utils.frameworks.Framework = <Framework.TENSORFLOW: 'tensorflow'>, device: 'Union[int, str, torch.device, None]' = None, meta: Optional[dict] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `loss_fn` | `Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]` |  |  |
| `sim_fn` | `Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `precompute_grads` | `bool` | `False` |  |
| `backend` | `alibi.utils.frameworks.Framework` | `<Framework.TENSORFLOW: 'tensorflow'>` |  |
| `device` | `Union[int, str, torch.device, None]` | `None` |  |
| `meta` | `Optional[dict]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `fit`

```python
fit(X_train: Union[numpy.ndarray, List[typing.Any]], Y_train: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

Fit the explainer. If ``self.precompute_grads == True`` then the gradients are precomputed and stored.

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

#### `reset_predictor`

```python
reset_predictor(predictor: Union[tensorflow.keras.Model, torch.nn.Module]) -> None
```

Resets the predictor to the given predictor.

Parameters
----------
predictor
    The new predictor to use.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |

**Returns**
- Type: `None`
