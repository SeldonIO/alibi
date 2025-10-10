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

### Constructor

```python
ProtoSelect(self, kernel_distance: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray], eps: float, lambda_penalty: Optional[float] = None, batch_size: int = 10000000000, preprocess_fn: Optional[Callable[[Union[list, numpy.ndarray]], numpy.ndarray]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `kernel_distance` | `Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray]` |  | Kernel distance to be used. Expected to support computation in batches. Given an input `x` of size `Nx x f1 x f2 x ...` and an input `y` of size `Ny x f1 x f2 x ...`, the kernel distance should return a kernel matrix of size `Nx x Ny`. |
| `eps` | `float` |  | Epsilon ball size. |
| `lambda_penalty` | `Optional[float]` | `None` | Penalty for each prototype. Encourages a lower number of prototypes to be selected. Corresponds to :math:`\lambda` in the paper notation. If not specified, the default value is set to `1 / N` where `N` is the size of the dataset to choose the prototype instances from, passed to the :py:meth:`alibi.prototypes.protoselect.ProtoSelect.fit` method. |
| `batch_size` | `int` | `10000000000` | Batch size to be used for kernel matrix computation. |
| `preprocess_fn` | `Optional[Callable[[.[typing.Union[list, numpy.ndarray]]], numpy.ndarray]]` | `None` | Preprocessing function used for kernel matrix computation. The preprocessing function takes the input as a `list` or a `numpy` array and transforms it into a `numpy` array which is then fed to the `kernel_distance` function. The use of `preprocess_fn` allows the method to be applied to any data modality. |
| `verbose` | `bool` | `False` | Whether to display progression bar while computing prototype points. |

### Methods

#### `fit`

```python
fit(X: Union[list, numpy.ndarray], y: Optional[numpy.ndarray] = None, Z: Union[list, numpy.ndarray, None] = None) -> alibi.prototypes.protoselect.ProtoSelect
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[list, numpy.ndarray]` |  | Dataset to be summarised. |
| `y` | `Optional[numpy.ndarray]` | `None` | Labels of the dataset `X` to be summarised. The labels are expected to be represented as integers `[0, 1, ..., L-1]`, where `L` is the number of classes in the dataset `X`. |
| `Z` | `Union[list, numpy.ndarray, None]` | `None` | Optional dataset to choose the prototypes from. If ``Z=None``, the prototypes will be selected from the dataset `X`. Otherwise, if `Z` is provided, the dataset to be summarised is still `X`, but it is summarised by prototypes belonging to the dataset `Z`. |

**Returns**
- Type: `alibi.prototypes.protoselect.ProtoSelect`

#### `summarise`

```python
summarise(num_prototypes: int = 1) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_prototypes` | `int` | `1` | Maximum number of prototypes to be selected. |

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
| `summary` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.prototypes.protoselect.ProtoSelect.summarise` method. |
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  | Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels. |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` | Optional preprocessor function. If ``preprocess_fn=None``, no preprocessing is applied. |
| `knn_kw` | `Optional[dict]` | `None` | Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be set automatically to 1, but the `metric` has to be specified according to the kernel distance used. If the `metric` is not specified, it will be set by default to ``'euclidean'``. See parameters description: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html |

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
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  | Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels. |
| `protoset` | `Optional[Tuple[numpy.ndarray]]` | `None` | Tuple, `(Z, )`, consisting of the dataset to choose the prototypes from. If `Z` is not provided (i.e., ``protoset=None``), the prototypes will be selected from the training dataset `X`. Otherwise, if `Z` is provided, the dataset to be summarised is still `X`, but it is summarised by prototypes belonging to the dataset `Z`. Note that the argument is passed as a tuple with a single element for consistency reasons. |
| `valset` | `Optional[Tuple[numpy.ndarray, numpy.ndarray]]` | `None` | Optional tuple `(X_val, y_val)` consisting of validation data instances with the corresponding validation labels. 1-KNN classifier is evaluated on the validation dataset to obtain the best epsilon radius. In case ``valset=None``, then `n-splits` cross-validation is performed on the `trainset`. |
| `num_prototypes` | `int` | `1` | The number of prototypes to be selected. |
| `eps_grid` | `Optional[numpy.ndarray]` | `None` | Optional grid of values to select the epsilon radius from. If not specified, the search grid is automatically proposed based on the inter-distances between `X` and `Z`. The distances are filtered by considering only values in between the `quantiles` values. The minimum and maximum distance values are used to define the range of values to search the epsilon radius. The interval is discretized in `grid_size` equidistant bins. |
| `quantiles` | `Optional[Tuple[float, float]]` | `None` | Quantiles, `(q_min, q_max)`, to be used to filter the range of values of the epsilon radius. The expected quantile values are in `[0, 1]` and clipped to `[0, 1]` if outside the range. See `eps_grid` for usage. If not specified, no filtering is applied. Only used if ``eps_grid=None``. |
| `grid_size` | `int` | `25` | The number of equidistant bins to be used to discretize the `eps_grid` automatically proposed interval. Only used if ``eps_grid=None``. |
| `n_splits` | `int` | `2` |  |
| `batch_size` | `int` | `10000000000` | Batch size to be used for kernel matrix computation. |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` | Preprocessing function to be applied to the data instance before applying the kernel. |
| `protoselect_kw` | `Optional[dict]` | `None` | Keyword arguments passed to :py:meth:`alibi.prototypes.protoselect.ProtoSelect.__init__`. |
| `knn_kw` | `Optional[dict]` | `None` | Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be set automatically to 1 and the `metric` will be set to ``'euclidean``. See parameters description: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html |
| `kfold_kw` | `Optional[dict]` | `None` | Keyword arguments passed to `sklearn.model_selection.KFold`. See parameters description: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html |

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
| `summary` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.prototypes.protoselect.ProtoSelect.summarise` method. |
| `trainset` | `Tuple[numpy.ndarray, numpy.ndarray]` |  | Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels. |
| `reducer` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | 2D reducer. Reduces the input feature representation to 2D. Note that the reducer operates directly on the input instances if ``preprocess_fn=None``. If the `preprocess_fn` is specified, the reducer will be called on the feature representation obtained after passing the input instances through the `preprocess_fn`. |
| `preprocess_fn` | `Optional[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` | `None` | Optional preprocessor function. If ``preprocess_fn=None``, no preprocessing is applied. |
| `knn_kw` | `Optional[dict]` | `None` | Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be set automatically to 1, but the `metric` has to be specified according to the kernel distance used. If the `metric` is not specified, it will be set by default to ``'euclidean'``. See parameters description: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html |
| `ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` | A `matplotlib` axes object to plot on. |
| `fig_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `fig.set` function. |
| `image_size` | `Tuple[int, int]` | `(28, 28)` | Shape to which the prototype images will be resized. A zoom of 1 will display the image having the shape `image_size`. |
| `zoom_lb` | `float` | `1.0` | Zoom lower bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`. |
| `zoom_ub` | `float` | `3.0` | Zoom upper bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`. |

**Returns**
- Type: `matplotlib.axes._axes.Axes`
