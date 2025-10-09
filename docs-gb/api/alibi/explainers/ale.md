# `alibi.explainers.ale`
## `ALE`

_Inherits from:_ `Explainer`, `ABC`, `Base`

### Constructor

```python
ALE(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], feature_names: Optional[List[str]] = None, target_names: Optional[List[str]] = None, check_feature_resolution: bool = True, low_resolution_threshold: int = 10, extrapolate_constant: bool = True, extrapolate_constant_perc: float = 10.0, extrapolate_constant_min: float = 0.1) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `target_names` | `Optional[List[str]]` | `None` |  |
| `check_feature_resolution` | `bool` | `True` |  |
| `low_resolution_threshold` | `int` | `10` |  |
| `extrapolate_constant` | `bool` | `True` |  |
| `extrapolate_constant_perc` | `float` | `10.0` |  |
| `extrapolate_constant_min` | `float` | `0.1` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[int]] = None, min_bin_points: int = 4, grid_points: Optional[Dict[int, numpy.ndarray]] = None) -> alibi.api.interfaces.Explanation
```

Calculate the ALE curves for each feature with respect to the dataset `X`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `features` | `Optional[List[int]]` | `None` |  |
| `min_bin_points` | `int` | `4` |  |
| `grid_points` | `Optional[Dict[int, numpy.ndarray]]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

## Functions
### `adaptive_grid`

```python
adaptive_grid(values: numpy.ndarray, min_bin_points: int = 1) -> Tuple[numpy.ndarray, int]
```

Find the optimal number of quantiles for the range of values so that each resulting bin

contains at least `min_bin_points`. Uses bisection.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |
| `min_bin_points` | `int` | `1` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, int]`

### `ale_num`

```python
ale_num(predictor: Callable, X: numpy.ndarray, feature: int, feature_grid_points: Optional[numpy.ndarray] = None, min_bin_points: int = 4, check_feature_resolution: bool = True, low_resolution_threshold: int = 10, extrapolate_constant: bool = True, extrapolate_constant_perc: float = 10.0, extrapolate_constant_min: float = 0.1) -> Tuple[numpy.ndarray, .Ellipsis]
```

Calculate the first order ALE curve for a numerical feature.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |
| `X` | `numpy.ndarray` |  |  |
| `feature` | `int` |  |  |
| `feature_grid_points` | `Optional[numpy.ndarray]` | `None` |  |
| `min_bin_points` | `int` | `4` |  |
| `check_feature_resolution` | `bool` | `True` |  |
| `low_resolution_threshold` | `int` | `10` |  |
| `extrapolate_constant` | `bool` | `True` |  |
| `extrapolate_constant_perc` | `float` | `10.0` |  |
| `extrapolate_constant_min` | `float` | `0.1` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, .Ellipsis]`

### `bisect_fun`

```python
bisect_fun(fun: Callable, target: float, lo: int, hi: int) -> int
```

Bisection algorithm for function evaluation with integer support.

Assumes the function is non-decreasing on the interval `[lo, hi]`.
Return an integer value v such that for all `x<v, fun(x)<target` and for all `x>=v, fun(x)>=target`.
This is equivalent to the library function `bisect.bisect_left` but for functions defined on integers.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fun` | `Callable` |  |  |
| `target` | `float` |  |  |
| `lo` | `int` |  |  |
| `hi` | `int` |  |  |

**Returns**
- Type: `int`

### `get_quantiles`

```python
get_quantiles(values: numpy.ndarray, num_quantiles: int = 11, interpolation = 'linear') -> numpy.ndarray
```

Calculate quantiles of values in an array.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |
| `num_quantiles` | `int` | `11` |  |
| `interpolation` |  | `'linear'` |  |

**Returns**
- Type: `numpy.ndarray`

### `minimum_satisfied`

```python
minimum_satisfied(values: numpy.ndarray, min_bin_points: int, n: int) -> int
```

Calculates whether the partition into bins induced by `n` quantiles

has the minimum number of points in each resulting bin.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |
| `min_bin_points` | `int` |  |  |
| `n` | `int` |  |  |

**Returns**
- Type: `int`

### `plot_ale`

```python
plot_ale(exp: alibi.api.interfaces.Explanation, features: Union[List[Union[str, int]], Literal[all]] = 'all', targets: Union[List[Union[str, int]], Literal[all]] = 'all', n_cols: int = 3, sharey: str = 'all', constant: bool = False, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, line_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> np.ndarray
```

Plot ALE curves on matplotlib axes.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  |  |
| `features` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` |  |
| `targets` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` |  |
| `n_cols` | `int` | `3` |  |
| `sharey` | `str` | `'all'` |  |
| `constant` | `bool` | `False` |  |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` |  |
| `line_kw` | `Optional[dict]` | `None` |  |
| `fig_kw` | `Optional[dict]` | `None` |  |

**Returns**
- Type: `np.ndarray`
