# `alibi.utils.distance`
## Functions
### `abdm`

```python
abdm(X: numpy.ndarray, cat_vars: dict, cat_vars_bin: dict = {})
```

Calculate the pair-wise distances between categories of a categorical variable using

the Association-Based Distance Metric based on Le et al (2005).
http://www.jaist.ac.jp/~bao/papers/N26.pdf

Parameters
----------
X
    Batch of arrays.
cat_vars
    Dict with as keys the categorical columns and as optional values
    the number of categories per categorical variable.
cat_vars_bin
    Dict with as keys the binned numerical columns and as optional values
    the number of bins per variable.

Returns
-------
Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `cat_vars` | `dict` |  |  |
| `cat_vars_bin` | `dict` | `{}` |  |

### `batch_compute_kernel_matrix`

```python
batch_compute_kernel_matrix(x: Union[list, numpy.ndarray], y: Union[list, numpy.ndarray], kernel: Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray], batch_size: int = 10000000000, preprocess_fn: Optional[Callable[[.[typing.Union[list, numpy.ndarray]]], numpy.ndarray]] = None) -> numpy.ndarray
```

Compute the kernel matrix between `x` and `y` by filling in blocks of size

`batch_size x batch_size` at a time.

Parameters
----------
x
    The first list/`numpy` array of data instances.
y
    The second list/`numpy` array of data instances.
kernel
    Kernel function to be used for kernel matrix computation.
batch_size
    Batch size to be used for each prediction.
preprocess_fn
    Optional preprocessing function for each batch.

Returns
-------
Kernel matrix in the form of a `numpy` array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray]` |  |  |
| `y` | `Union[list, numpy.ndarray]` |  |  |
| `kernel` | `Callable[[.[<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `batch_size` | `int` | `10000000000` |  |
| `preprocess_fn` | `Optional[Callable[[.[typing.Union[list, numpy.ndarray]]], numpy.ndarray]]` | `None` |  |

**Returns**
- Type: `numpy.ndarray`

### `cityblock_batch`

```python
cityblock_batch(X: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray
```

Calculate the L1 distances between a batch of arrays `X` and an array of the same shape `y`.

Parameters
----------
X
    Batch of arrays to calculate the distances from.
y
    Array to calculate the distance to.

Returns
-------
Array of distances from each array in `X` to `y`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `y` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `multidim_scaling`

```python
multidim_scaling(d_pair: dict, feature_range: Tuple[numpy.ndarray, numpy.ndarray], n_components: int = 2, use_metric: bool = True, standardize_cat_vars: bool = True, smooth: float = 1.0, center: bool = True, update_feature_range: bool = True) -> Tuple[dict, tuple]
```

Apply multidimensional scaling to pairwise distance matrices.

Parameters
----------
d_pair
    Dict with as keys the column index of the categorical variables and as values
    a pairwise distance matrix for the categories of the variable.
feature_range
    Tuple with `min` and `max` ranges to allow for perturbed instances. `Min` and `max` ranges are
    `numpy` arrays with dimension (`1 x nb of features`).
n_components
    Number of dimensions in which to immerse the dissimilarities.
use_metric
    If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
standardize_cat_vars
    Standardize numerical values of categorical variables if ``True``.
smooth
    Smoothing exponent between 0 and 1 for the distances. Lower values than 1 will smooth the difference in
    distance metric between different features.
center
    Whether to center the scaled distance measures. If ``False``, the min distance for each feature
    except for the feature with the highest raw max distance will be the lower bound of the
    feature range, but the upper bound will be below the max feature range.
update_feature_range
    Update feature range with scaled values.

Returns
-------
Dict with multidimensional scaled version of pairwise distance matrices.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `d_pair` | `dict` |  |  |
| `feature_range` | `Tuple[numpy.ndarray, numpy.ndarray]` |  |  |
| `n_components` | `int` | `2` |  |
| `use_metric` | `bool` | `True` |  |
| `standardize_cat_vars` | `bool` | `True` |  |
| `smooth` | `float` | `1.0` |  |
| `center` | `bool` | `True` |  |
| `update_feature_range` | `bool` | `True` |  |

**Returns**
- Type: `Tuple[dict, tuple]`

### `mvdm`

```python
mvdm(X: numpy.ndarray, y: numpy.ndarray, cat_vars: dict, alpha: int = 1) -> Dict[int, numpy.ndarray]
```

Calculate the pair-wise distances between categories of a categorical variable using

the Modified Value Difference Measure based on Cost et al (1993).
https://link.springer.com/article/10.1023/A:1022664626993

Parameters
----------
X
    Batch of arrays.
y
    Batch of labels or predictions.
cat_vars
    Dict with as keys the categorical columns and as optional values
    the number of categories per categorical variable.
alpha
    Power of absolute difference between conditional probabilities.

Returns
-------
Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `y` | `numpy.ndarray` |  |  |
| `cat_vars` | `dict` |  |  |
| `alpha` | `int` | `1` |  |

**Returns**
- Type: `Dict[int, numpy.ndarray]`

### `squared_pairwise_distance`

```python
squared_pairwise_distance(x: numpy.ndarray, y: numpy.ndarray, a_min: float = 1e-07, a_max: float = 1e+30) -> numpy.ndarray
```

`numpy` pairwise squared Euclidean distance between samples `x` and `y`.

Parameters
----------
x
    A batch of instances of shape `Nx x features`.
y
    A batch of instances of shape `Ny x features`.
a_min
    Lower bound to clip distance values.
a_max
    Upper bound to clip distance values.

Returns
-------
Pairwise squared Euclidean distance `Nx x Ny`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  |  |
| `y` | `numpy.ndarray` |  |  |
| `a_min` | `float` | `1e-07` |  |
| `a_max` | `float` | `1e+30` |  |

**Returns**
- Type: `numpy.ndarray`
