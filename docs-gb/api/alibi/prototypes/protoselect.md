# `alibi.prototypes.protoselect`
## Constants
### `DEFAULT_DATA_PROTOSELECT`
```python
DEFAULT_DATA_PROTOSELECT: dict = {'prototypes': None, 'prototype_indices': None, 'prototype_labels': None}
```

### `DEFAULT_META_PROTOSELECT`
```python
DEFAULT_META_PROTOSELECT: dict = {'name': None, 'type': ['data'], 'explanation': ['global'], 'params': {}, 've...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.prototypes.protoselect (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

## `ProtoSelect`

_Inherits from:_ `Summariser`, `FitMixin`, `ABC`, `Base`

Base class for prototype algorithms from :py:mod:`alibi.prototypes`.

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

### `fit`

```python
fit(X: Union[list, numpy.ndarray], y: Optional[numpy.ndarray] = None, Z: Union[list, numpy.ndarray, None] = None) -> alibi.prototypes.protoselect.ProtoSelect
```

Fit the summariser. This step forms the kernel matrix in memory which has a shape of `NX x NX`,

where `NX` is  the number of instances in `X`, if the optional dataset `Z` is not provided. Otherwise, if
the optional dataset `Z` is provided, the kernel matrix has a shape of `NZ x NX`, where `NZ` is the
number of instances in `Z`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[list, numpy.ndarray]` |  |  |
| `y` | `Optional[numpy.ndarray]` | `None` |  |
| `Z` | `Union[list, numpy.ndarray, None]` | `None` |  |

**Returns**
- Type: `alibi.prototypes.protoselect.ProtoSelect`

### `summarise`

```python
summarise(num_prototypes: int = 1) -> alibi.api.interfaces.Explanation
```

Searches for the requested number of prototypes. Note that the algorithm can return a lower number of

prototypes than the requested one. To increase the number of prototypes, reduce the epsilon-ball radius
(`eps`), and the penalty for adding a prototype (`lambda_penalty`).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_prototypes` | `int` | `1` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `compute_prototype_importances`

```python
compute_prototype_importances(summary: alibi.api.interfaces.Explanation, trainset: Tuple[numpy.ndarray, numpy.ndarray], preprocess_fn: Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]] = None, knn_kw: Optional[dict] = None) -> Dict[str, Optional[numpy.ndarray]]
```

Computes the importance of each prototype. The importance of a prototype is the number of assigned

training instances correctly classified according to the 1-KNN classifier
(Bien and Tibshirani (2012): https://arxiv.org/abs/1202.5933).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `summary` | `alibi.api.interfaces.Explanation` |  |  |
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  |  |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` |  |
| `knn_kw` | `Optional[dict]` | `None` |  |

**Returns**
- Type: `Dict[str, Optional[numpy.ndarray]]`

### `cv_protoselect_euclidean`

```python
cv_protoselect_euclidean(trainset: Tuple[numpy.ndarray, numpy.ndarray], protoset: Optional[Tuple[numpy.ndarray]] = None, valset: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None, num_prototypes: int = 1, eps_grid: Optional[numpy.ndarray] = None, quantiles: Optional[Tuple[float, float]] = None, grid_size: int = 25, n_splits: int = 2, batch_size: int = 10000000000, preprocess_fn: Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]] = None, protoselect_kw: Optional[dict] = None, knn_kw: Optional[dict] = None, kfold_kw: Optional[dict] = None) -> dict
```

Cross-validation parameter selection for `ProtoSelect` with Euclidean distance. The method computes

the best epsilon radius.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  |  |
| `protoset` | `Optional[Tuple[numpy.ndarray]]` | `None` |  |
| `valset` | `Optional[Tuple[numpy.ndarray, numpy.ndarray]]` | `None` |  |
| `num_prototypes` | `int` | `1` |  |
| `eps_grid` | `Optional[numpy.ndarray]` | `None` |  |
| `quantiles` | `Optional[Tuple[float, float]]` | `None` |  |
| `grid_size` | `int` | `25` |  |
| `n_splits` | `int` | `2` |  |
| `batch_size` | `int` | `10000000000` |  |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` |  |
| `protoselect_kw` | `Optional[dict]` | `None` |  |
| `knn_kw` | `Optional[dict]` | `None` |  |
| `kfold_kw` | `Optional[dict]` | `None` |  |

**Returns**
- Type: `dict`

### `visualize_image_prototypes`

```python
visualize_image_prototypes(summary: alibi.api.interfaces.Explanation, trainset: Tuple[numpy.ndarray, numpy.ndarray], reducer: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], preprocess_fn: Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]] = None, knn_kw: Optional[dict] = None, ax: Optional[matplotlib.axes._axes.Axes] = None, fig_kw: Optional[dict] = None, image_size: Tuple[int, int] = (28, 28), zoom_lb: float = 1.0, zoom_ub: float = 3.0) -> matplotlib.axes._axes.Axes
```

Plot the images of the prototypes at the location given by the `reducer` representation.

The size of each prototype is proportional to the logarithm of the number of assigned training instances correctly
classified according to the 1-KNN classifier (Bien and Tibshirani (2012): https://arxiv.org/abs/1202.5933).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `summary` | `alibi.api.interfaces.Explanation` |  |  |
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  |  |
| `reducer` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` |  |
| `knn_kw` | `Optional[dict]` | `None` |  |
| `ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` |  |
| `fig_kw` | `Optional[dict]` | `None` |  |
| `image_size` | `Tuple[int, int]` | `(28, 28)` |  |
| `zoom_lb` | `float` | `1.0` |  |
| `zoom_ub` | `float` | `3.0` |  |

**Returns**
- Type: `matplotlib.axes._axes.Axes`
