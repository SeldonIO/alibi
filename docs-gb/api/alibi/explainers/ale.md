# `alibi.explainers.ale`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `DEFAULT_META_ALE`
```python
DEFAULT_META_ALE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `DEFAULT_DATA_ALE`
```python
DEFAULT_DATA_ALE: dict = {'ale_values': [], 'constant_value': None, 'ale0': [], 'feature_values': [], ...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.ale (WARNING)>
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

## `ALE`

_Inherits from:_ `Explainer`, `ABC`, `Base`

### Constructor

```python
ALE(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], feature_names: Optional[List[str]] = None, target_names: Optional[List[str]] = None, check_feature_resolution: bool = True, low_resolution_threshold: int = 10, extrapolate_constant: bool = True, extrapolate_constant_perc: float = 10.0, extrapolate_constant_min: float = 0.1) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes in an `N x F` array as input and outputs an `N x T` array (`N` - number of data points, `F` - number of features, `T` - number of outputs/targets (e.g. 1 for single output regression, >=2 for classification)). |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results. |
| `target_names` | `Optional[List[str]]` | `None` | A list of target/output names used for displaying results. |
| `check_feature_resolution` | `bool` | `True` | If ``True``, the number of unique values is calculated for each feature and if it is less than `low_resolution_threshold` then the feature values are used for grid-points instead of quantiles. This may increase the runtime of the algorithm for large datasets. Only used for features without custom grid-points specified in :py:meth:`alibi.explainers.ale.ALE.explain`. |
| `low_resolution_threshold` | `int` | `10` | If a feature has at most this many unique values, these are used as the grid points instead of quantiles. This is to avoid situations when the quantile algorithm returns quantiles between discrete values which can result in jumps in the ALE plot obscuring the true effect. Only used if `check_feature_resolution` is ``True`` and for features without custom grid-points specified in :py:meth:`alibi.explainers.ale.ALE.explain`. |
| `extrapolate_constant` | `bool` | `True` | If a feature is constant, only one quantile exists where all the data points lie. In this case the ALE value at that point is zero, however this may be misleading if the feature does have an effect on the model. If this parameter is set to ``True``, the ALE values are calculated on an interval surrounding the constant value. The interval length is controlled by the `extrapolate_constant_perc` and `extrapolate_constant_min` arguments. |
| `extrapolate_constant_perc` | `float` | `10.0` | Percentage by which to extrapolate a constant feature value to create an interval for ALE calculation. If `q` is the constant feature value, creates an interval `[q - q/extrapolate_constant_perc, q + q/extrapolate_constant_perc]` for which ALE is calculated. Only relevant if `extrapolate_constant` is set to ``True``. |
| `extrapolate_constant_min` | `float` | `0.1` | Controls the minimum extrapolation length for constant features. An interval constructed for constant features is guaranteed to be `2 x extrapolate_constant_min` wide centered on the feature value. This allows for capturing model behaviour around constant features which have small value so that `extrapolate_constant_perc` is not so helpful. Only relevant if `extrapolate_constant` is set to ``True``. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[int]] = None, min_bin_points: int = 4, grid_points: Optional[Dict[int, numpy.ndarray]] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | An `N x F` tabular dataset used to calculate the ALE curves. This is typically the training dataset or a representative sample. |
| `features` | `Optional[List[int]]` | `None` | Features for which to calculate ALE. |
| `min_bin_points` | `int` | `4` | Minimum number of points each discretized interval should contain to ensure more precise ALE estimation. Only relevant for adaptive grid points (i.e., features without an entry in the `grid_points` dictionary). |
| `grid_points` | `Optional[Dict[int, numpy.ndarray]]` | `None` | Custom grid points. Must be a `dict` where the keys are features indices and the values are monotonically increasing `numpy` arrays defining the grid points for each feature. See the :ref:`Notes<Notes ALE explain>` section for the default behavior when potential edge-cases arise when using grid-points. If no grid points are specified (i.e. the feature is missing from the `grid_points` dictionary), deciles discretization is used instead. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New predictor function. |

**Returns**
- Type: `None`

## Functions
### `adaptive_grid`

```python
adaptive_grid(values: numpy.ndarray, min_bin_points: int = 1) -> Tuple[numpy.ndarray, int]
```

Find the optimal number of quantiles for the range of values so that each resulting bin
contains at least `min_bin_points`. Uses bisection.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  | Array of feature values. |
| `min_bin_points` | `int` | `1` | Minimum number of points each discretized interval should contain to ensure more precise ALE estimation. |

**Returns**
- Type: `Tuple[numpy.ndarray, int]`

### `ale_num`

```python
ale_num(predictor: Callable, X: numpy.ndarray, feature: int, feature_grid_points: Optional[numpy.ndarray] = None, min_bin_points: int = 4, check_feature_resolution: bool = True, low_resolution_threshold: int = 10, extrapolate_constant: bool = True, extrapolate_constant_perc: float = 10.0, extrapolate_constant_min: float = 0.1) -> Tuple[numpy.ndarray, .Ellipsis]
```

Calculate the first order ALE curve for a numerical feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | Model prediction function. |
| `X` | `numpy.ndarray` |  | Dataset for which ALE curves are computed. |
| `feature` | `int` |  | Index of the numerical feature for which to calculate ALE. |
| `feature_grid_points` | `Optional[numpy.ndarray]` | `None` | Custom grid points. An `numpy` array defining the grid points for the given features. |
| `min_bin_points` | `int` | `4` | Minimum number of points each discretized interval should contain to ensure more precise ALE estimation. Only relevant for adaptive grid points (i.e., feature for which ``feature_grid_points=None``). |
| `check_feature_resolution` | `bool` | `True` | Refer to :class:`ALE` documentation. |
| `low_resolution_threshold` | `int` | `10` | Refer to :class:`ALE` documentation. |
| `extrapolate_constant` | `bool` | `True` | Refer to :class:`ALE` documentation. |
| `extrapolate_constant_perc` | `float` | `10.0` | Refer to :class:`ALE` documentation. |
| `extrapolate_constant_min` | `float` | `0.1` | Refer to :class:`ALE` documentation. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fun` | `Callable` |  | A function defined on integers in the range `[lo, hi]` and returning floats. |
| `target` | `float` |  | Target value to be searched for. |
| `lo` | `int` |  | Lower bound of the domain. |
| `hi` | `int` |  | Upper bound of the domain. |

**Returns**
- Type: `int`

### `get_quantiles`

```python
get_quantiles(values: numpy.ndarray, num_quantiles: int = 11, interpolation = 'linear') -> numpy.ndarray
```

Calculate quantiles of values in an array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  | Array of values. |
| `num_quantiles` | `int` | `11` | Number of quantiles to calculate. |
| `interpolation` |  | `'linear'` |  |

**Returns**
- Type: `numpy.ndarray`

### `minimum_satisfied`

```python
minimum_satisfied(values: numpy.ndarray, min_bin_points: int, n: int) -> int
```

Calculates whether the partition into bins induced by `n` quantiles
has the minimum number of points in each resulting bin.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  | Array of feature values. |
| `min_bin_points` | `int` |  | Minimum number of points each discretized interval needs to contain. |
| `n` | `int` |  | Number of quantiles. |

**Returns**
- Type: `int`

### `plot_ale`

```python
plot_ale(exp: alibi.api.interfaces.Explanation, features: Union[List[Union[str, int]], Literal[all]] = 'all', targets: Union[List[Union[str, int]], Literal[all]] = 'all', n_cols: int = 3, sharey: str = 'all', constant: bool = False, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, line_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> np.ndarray
```

Plot ALE curves on matplotlib axes.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.ale.ALE.explain` method. |
| `features` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` | A list of features for which to plot the ALE curves or ``'all'`` for all features. Can be a mix of integers denoting feature index or strings denoting entries in `exp.feature_names`. Defaults to ``'all'``. |
| `targets` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` | A list of targets for which to plot the ALE curves or ``'all'`` for all targets. Can be a mix of integers denoting target index or strings denoting entries in `exp.target_names`. Defaults to ``'all'``. |
| `n_cols` | `int` | `3` | Number of columns to organize the resulting plot into. |
| `sharey` | `str` | `'all'` | A parameter specifying whether the y-axis of the ALE curves should be on the same scale for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``. |
| `constant` | `bool` | `False` | A parameter specifying whether the constant zeroth order effects should be added to the ALE first order effects. |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` | A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on. |
| `line_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `plt.plot` function. |
| `fig_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `fig.set` function. |

**Returns**
- Type: `np.ndarray`
