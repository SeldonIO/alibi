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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  |  |
| `y` | `numpy.ndarray` |  |  |
| `a_min` | `float` | `1e-07` |  |
| `a_max` | `float` | `1e+30` |  |

**Returns**
- Type: `numpy.ndarray`
