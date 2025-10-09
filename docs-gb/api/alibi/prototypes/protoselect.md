# `alibi.prototypes.protoselect`
## `ProtoSelect`

_Inherits from:_ `Summariser`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
ProtoSelect(self, kernel_distance: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray], eps: float, lambda_penalty: Optional[float] = None, batch_size: int = 10000000000, preprocess_fn: Optional[Callable[[Union[list, numpy.ndarray]], numpy.ndarray]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `kernel_distance` | `Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `eps` | `float` |  |  |
| `lambda_penalty` | `Optional[float]` | `None` |  |
| `batch_size` | `int` | `10000000000` |  |
| `preprocess_fn` | `Optional[Callable[[.[typing.Union[list, numpy.ndarray]]], numpy.ndarray]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `fit`

```python
fit(X: Union[list, numpy.ndarray], y: Optional[numpy.ndarray] = None, Z: Union[list, numpy.ndarray, None] = None) -> alibi.prototypes.protoselect.ProtoSelect
```

Fit the summariser. This step forms the kernel matrix in memory which has a shape of `NX x NX`,

where `NX` is  the number of instances in `X`, if the optional dataset `Z` is not provided. Otherwise, if
the optional dataset `Z` is provided, the kernel matrix has a shape of `NZ x NX`, where `NZ` is the
number of instances in `Z`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[list, numpy.ndarray]` |  |  |
| `y` | `Optional[numpy.ndarray]` | `None` |  |
| `Z` | `Union[list, numpy.ndarray, None]` | `None` |  |

**Returns**
- Type: `alibi.prototypes.protoselect.ProtoSelect`

#### `summarise`

```python
summarise(num_prototypes: int = 1) -> alibi.api.interfaces.Explanation
```

Searches for the requested number of prototypes. Note that the algorithm can return a lower number of

prototypes than the requested one. To increase the number of prototypes, reduce the epsilon-ball radius
(`eps`), and the penalty for adding a prototype (`lambda_penalty`).

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_prototypes` | `int` | `1` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`
