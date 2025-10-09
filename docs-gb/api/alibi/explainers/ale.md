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
----------
X
    An `N x F` tabular dataset used to calculate the ALE curves. This is typically the training dataset
    or a representative sample.
features
    Features for which to calculate ALE.
min_bin_points
    Minimum number of points each discretized interval should contain to ensure more precise
    ALE estimation. Only relevant for adaptive grid points (i.e., features without an entry in the
    `grid_points` dictionary).
grid_points
    Custom grid points. Must be a `dict` where the keys are features indices and the values are
    monotonically increasing `numpy` arrays defining the grid points for each feature.
    See the :ref:`Notes<Notes ALE explain>` section for the default behavior when potential edge-cases arise
    when using grid-points. If no grid points are specified (i.e. the feature is missing from the `grid_points`
    dictionary), deciles discretization is used instead.

Returns
-------
explanation
    An `Explanation` object containing the data and the metadata of the calculated ALE curves.
    See usage at `ALE examples`_ for details.

    .. _ALE examples:
        https://docs.seldon.io/projects/alibi/en/latest/methods/ALE.html

Notes
-----
.. _Notes ALE explain:

Consider `f` to be a feature of interest. We denote possible feature values of `f` by `X` (i.e. the values
from the dataset column corresponding to feature `f`), by `O` a user-specified grid-point value, and by
`(X|O)` an overlap between a grid-point and a feature value. We can encounter the following edge-cases:

 - Grid points outside the feature range. Consider the following example: `O O O X X O X O X O O`,         where 3 grid-points are smaller than the minimum value in `f`, and 2 grid-points are larger than the maximum         value in `f`. The empty leading and ending bins are removed. The grid-points considered
will be: `O X X O X O X O`.

 - Grid points that do not cover the entire feature range. Consider the following example:         `X X O X X O X O X X X X X`. Two auxiliary grid-points are added which correspond the value of the minimum         and maximum value of feature `f`. The grid-points considered will be: `(O|X) X O X X O X O X X X X (X|O)`.

 - Grid points that do not contain any values in between. Consider the following example:         `(O|X) X X O O O X O X O O (X|O)`. The intervals which do not contain any feature values are removed/merged.         The grid-points considered will be: `(O|X) X X O X O X O (X|O)`.

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
----------
predictor
    New predictor function.

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
----------
values
    Array of feature values.
min_bin_points
    Minimum number of points each discretized interval should contain to ensure more precise
    ALE estimation.

Returns
-------
q
    Unique quantiles.
num_quantiles
    Number of non-unique quantiles the feature array was subdivided into.

Notes
-----
This is a heuristic procedure since the bisection algorithm is applied
to a function which is not monotonic. This will not necessarily find the
maximum number of bins the interval can be subdivided into to satisfy
the minimum number of points in each resulting bin.

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
----------
predictor
    Model prediction function.
X
    Dataset for which ALE curves are computed.
feature
    Index of the numerical feature for which to calculate ALE.
feature_grid_points
    Custom grid points. An `numpy` array defining the grid points for the given features.
min_bin_points
    Minimum number of points each discretized interval should contain to ensure more precise
    ALE estimation. Only relevant for adaptive grid points (i.e., feature for which ``feature_grid_points=None``).
check_feature_resolution
    Refer to :class:`ALE` documentation.
low_resolution_threshold
    Refer to :class:`ALE` documentation.
extrapolate_constant
    Refer to :class:`ALE` documentation.
extrapolate_constant_perc
    Refer to :class:`ALE` documentation.
extrapolate_constant_min
    Refer to :class:`ALE` documentation.

Returns
-------
fvals
    Array of quantiles or custom grid-points of the input values.
ale
    ALE values for each feature at each of the points in `fvals`.
ale0
    The constant offset used to center the ALE curves.

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
----------
fun
    A function defined on integers in the range `[lo, hi]` and returning floats.
target
    Target value to be searched for.
lo
    Lower bound of the domain.
hi
    Upper bound of the domain.

Returns
-------
Integer index.

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
----------
values
    Array of values.
num_quantiles
    Number of quantiles to calculate.

Returns
-------
Array of quantiles of the input values.

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
----------
values
    Array of feature values.
min_bin_points
    Minimum number of points each discretized interval needs to contain.
n
    Number of quantiles.

Returns
-------
Integer encoded boolean with 1 - each bin has at least `min_bin_points` and 0 otherwise.

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
----------
exp
    An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.ale.ALE.explain` method.
features
    A list of features for which to plot the ALE curves or ``'all'`` for all features.
    Can be a mix of integers denoting feature index or strings denoting entries in
    `exp.feature_names`. Defaults to ``'all'``.
targets
    A list of targets for which to plot the ALE curves or ``'all'`` for all targets.
    Can be a mix of integers denoting target index or strings denoting entries in
    `exp.target_names`. Defaults to ``'all'``.
n_cols
    Number of columns to organize the resulting plot into.
sharey
    A parameter specifying whether the y-axis of the ALE curves should be on the same scale
    for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``.
constant
    A parameter specifying whether the constant zeroth order effects should be added to the
    ALE first order effects.
ax
    A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
line_kw
    Keyword arguments passed to the `plt.plot` function.
fig_kw
    Keyword arguments passed to the `fig.set` function.

Returns
-------
An array of `matplotlib` axes with the resulting ALE plots.

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
