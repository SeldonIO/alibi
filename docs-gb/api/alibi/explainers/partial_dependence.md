# `alibi.explainers.partial_dependence`
## Constants
### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD: dict = {'feature_deciles': None, 'pd_values': None, 'ice_values': None, 'feature_val...
```

### `DEFAULT_META_PD`
```python
DEFAULT_META_PD: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.partial_dependence (WARNING)>
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

## `Kind`

_Inherits from:_ `str`, `Enum`

Enumeration of supported kind.

## `PartialDependence`

_Inherits from:_ `PartialDependenceBase`, `Explainer`, `ABC`, `Base`

Black-box implementation of partial dependence for tabular datasets.
Supports multiple feature interactions.

### Constructor

```python
PartialDependence(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], feature_names: Optional[List[str]] = None, categorical_names: Optional[Dict[int, List[str]]] = None, target_names: Optional[List[str]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A prediction function which receives as input a `numpy` array of size `N x F` and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances, `F` is the number of features and `T` is the number of targets. |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. Necessary to identify the categorical features in the dataset. An example for `categorical_names` would be:: category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]} |
| `target_names` | `Optional[List[str]]` | `None` | A list of target/output names used for displaying results. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, int]]]] = None, kind: Literal[average, individual, both] = 'average', percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `features` | `Optional[List[Union[int, Tuple[int, int]]]]` | `None` |  |
| `kind` | `Literal[average, individual, both]` | `'average'` |  |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` |  |
| `grid_resolution` | `int` | `100` |  |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## `PartialDependenceBase`

_Inherits from:_ `Explainer`, `ABC`, `Base`

### Constructor

```python
PartialDependenceBase(self, predictor: Union[sklearn.base.BaseEstimator, Callable[[numpy.ndarray], numpy.ndarray]], feature_names: Optional[List[str]] = None, categorical_names: Optional[Dict[int, List[str]]] = None, target_names: Optional[List[str]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[sklearn.base.BaseEstimator, Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` |  | A `sklearn` estimator or a prediction function which receives as input a `numpy` array of size `N x F` and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances, `F` is the number of features and `T` is the number of targets. |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. Necessary to identify the categorical features in the dataset. An example for `categorical_names` would be:: category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]} |
| `target_names` | `Optional[List[str]]` | `None` | A list of target/output names used for displaying results. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, int]]]] = None, kind: Literal[average, individual, both] = 'average', percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the training dataset or a representative sample. |
| `features` | `Optional[List[Union[int, Tuple[int, int]]]]` | `None` | An optional list of features or tuples of features for which to calculate the partial dependence. If not provided, the partial dependence will be computed for every single features in the dataset. Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively. |
| `kind` | `Literal[average, individual, both]` | `'average'` | If set to ``'average'``, then only the partial dependence (PD) averaged across all samples from the dataset is returned. If set to ``'individual'``, then only the individual conditional expectation (ICE) is returned for each data point from the dataset. Otherwise, if set to ``'both'``, then both the PD and the ICE are returned. |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` | Lower and upper percentiles used to limit the feature values to potentially remove outliers from low-density regions. Note that for features with not many data points with large/low values, the PD estimates are less reliable in those extreme regions. The values must be in [0, 1]. Only used with `grid_resolution`. |
| `grid_resolution` | `int` | `100` | Number of equidistant points to split the range of each target feature. Only applies if the number of unique values of a target feature in the reference dataset `X` is greater than the `grid_resolution` value. For example, consider a case where a feature can take the following values: ``[0.1, 0.3, 0.35, 0.351, 0.4, 0.41, 0.44, ..., 0.5, 0.54, 0.56, 0.6, 0.65, 0.7, 0.9]``, and we are not interested in evaluating the marginal effect at every single point as it can become computationally costly (assume hundreds/thousands of points) without providing any additional information for nearby points (e.g., 0.35 and 351). By setting ``grid_resolution=5``, the marginal effect is computed for the values ``[0.1, 0.3, 0.5, 0.7, 0.9]`` instead, which is less computationally demanding and can provide similar insights regarding the model's behaviour. Note that the extreme values of the grid can be controlled using the `percentiles` argument. |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` | Custom grid points. Must be a `dict` where the keys are the target features indices and the values are monotonically increasing arrays defining the grid points for a numerical feature, and a subset of categorical feature values for a categorical feature. If the `grid_points` are not specified, then the grid will be constructed based on the unique target feature values available in the dataset `X`, or based on the `grid_resolution` and `percentiles` (check `grid_resolution` to see when it applies). For categorical features, the corresponding value in the `grid_points` can be specified either as array of strings or array of integers corresponding the label encodings. Note that the label encoding must match the ordering of the values provided in the `categorical_names`. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], sklearn.base.BaseEstimator]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], sklearn.base.BaseEstimator]` |  | New predictor function or tree-based `sklearn` estimator. |

**Returns**
- Type: `None`

## `TreePartialDependence`

_Inherits from:_ `PartialDependenceBase`, `Explainer`, `ABC`, `Base`

Tree-based model `sklearn`  implementation of the partial dependence for tabular datasets.
Supports multiple feature interactions. This method is faster than the general black-box implementation
but is only supported by some tree-based estimators. The computation is based on a weighted tree traversal.
For more details on the computation, check the `sklearn documentation page`_. The supported `sklearn`
models are: `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`,
`HistGradientBoostingRegressor`, `HistGradientBoostingRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`.

.. _sklearn documentation page:
        https://scikit-learn.org/stable/modules/partial_dependence.html#computation-methods

### Constructor

```python
TreePartialDependence(self, predictor: sklearn.base.BaseEstimator, feature_names: Optional[List[str]] = None, categorical_names: Optional[Dict[int, List[str]]] = None, target_names: Optional[List[str]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `sklearn.base.BaseEstimator` |  | A tree-based `sklearn` estimator. |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. Necessary to identify the categorical features in the dataset. An example for `categorical_names` would be:: category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]} |
| `target_names` | `Optional[List[str]]` | `None` | A list of target/output names used for displaying results. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, int]]]] = None, percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the training dataset or a representative sample. |
| `features` | `Optional[List[Union[int, Tuple[int, int]]]]` | `None` | An optional list of features or tuples of features for which to calculate the partial dependence. If not provided, the partial dependence will be computed for every single features in the dataset. Some example for `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``, where ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively. |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` | Lower and upper percentiles used to limit the feature values to potentially remove outliers from low-density regions. Note that for features with not many data points with large/low values, the PD estimates are less reliable in those extreme regions. The values must be in [0, 1]. Only used with `grid_resolution`. |
| `grid_resolution` | `int` | `100` | Number of equidistant points to split the range of each target feature. Only applies if the number of unique values of a target feature in the reference dataset `X` is greater than the `grid_resolution` value. For example, consider a case where a feature can take the following values: ``[0.1, 0.3, 0.35, 0.351, 0.4, 0.41, 0.44, ..., 0.5, 0.54, 0.56, 0.6, 0.65, 0.7, 0.9]``, and we are not interested in evaluating the marginal effect at every single point as it can become computationally costly (assume hundreds/thousands of points) without providing any additional information for nearby points (e.g., 0.35 and 351). By setting ``grid_resolution=5``, the marginal effect is computed for the values ``[0.1, 0.3, 0.5, 0.7, 0.9]`` instead, which is less computationally demanding and can provide similar insights regarding the model's behaviour. Note that the extreme values of the grid can be controlled using the `percentiles` argument. |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` | Custom grid points. Must be a `dict` where the keys are the target features indices and the values are monotonically increasing arrays defining the grid points for a numerical feature, and a subset of categorical feature values for a categorical feature. If the `grid_points` are not specified, then the grid will be constructed based on the unique target feature values available in the dataset `X`, or based on the `grid_resolution` and `percentiles` (check `grid_resolution` to see when it applies). For categorical features, the corresponding value in the `grid_points` can be specified either as array of strings or array of integers corresponding the label encodings. Note that the label encoding must match the ordering of the values provided in the `categorical_names`. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `plot_pd`

```python
plot_pd(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', target: Union[str, int] = 0, n_cols: int = 3, n_ice: Union[Literal[all], int, List[int]] = 100, center: bool = False, pd_limits: Optional[Tuple[float, float]] = None, levels: int = 8, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, sharey: Optional[Literal[all, row]] = 'all', pd_num_kw: Optional[dict] = None, ice_num_kw: Optional[dict] = None, pd_cat_kw: Optional[dict] = None, ice_cat_kw: Optional[dict] = None, pd_num_num_kw: Optional[dict] = None, pd_num_cat_kw: Optional[dict] = None, pd_cat_cat_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> np.ndarray
```

Plot partial dependence curves on matplotlib axes.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.partial_dependence.PartialDependence.explain` method. |
| `features` | `Union[List[int], Literal[all]]` | `'all'` | A list of features entries in the `exp.data['feature_names']` to plot the partial dependence curves for, or ``'all'`` to plot all the explained feature or tuples of features. This includes tuples of features. For example, if ``exp.data['feature_names'] = ['temp', 'hum', ('temp', 'windspeed')]`` and we want to plot the partial dependence only for the ``'temp'`` and ``('temp', 'windspeed')``, then we would set ``features=[0, 2]``. Defaults to ``'all'``. |
| `target` | `Union[str, int]` | `0` | The target name or index for which to plot the partial dependence (PD) curves. Can be a mix of integers denoting target index or strings denoting entries in `exp.meta['params']['target_names']`. |
| `n_cols` | `int` | `3` | Number of columns to organize the resulting plot into. |
| `n_ice` | `Union[Literal[all], int, List[int]]` | `100` | Number of ICE plots to be displayed. Can be - a string taking the value ``'all'`` to display the ICE curves for every instance in the reference dataset. - an integer for which `n_ice` instances from the reference dataset will be sampled uniformly at random to          display their ICE curves. - a list of integers, where each integer represents an index of an instance in the reference dataset to          display their ICE curves. |
| `center` | `bool` | `False` | Boolean flag to center the individual conditional expectation (ICE) curves. As mentioned in `Goldstein et al. (2014)`_, the heterogeneity in the model can be difficult to discern when the intercepts of the ICE curves cover a wide range. Centering the ICE curves removes the level effects and helps to visualise the heterogeneous effect. .. _Goldstein et al. (2014): https://arxiv.org/abs/1309.6392 |
| `pd_limits` | `Optional[Tuple[float, float]]` | `None` | Minimum and maximum y-limits for all the one-way PD plots. If ``None`` will be automatically inferred. |
| `levels` | `int` | `8` | Number of levels in the contour plot. |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` | A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on. |
| `sharey` | `Optional[Literal[all, row]]` | `'all'` | A parameter specifying whether the y-axis of the PD and ICE curves should be on the same scale for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``. |
| `pd_num_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a numerical feature. |
| `ice_num_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the ICE for a numerical feature. |
| `pd_cat_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a categorical feature. |
| `ice_cat_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the ICE for a categorical feature. |
| `pd_num_num_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.contourf`_ function when plotting the PD for two numerical features. |
| `pd_num_cat_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function when plotting the PD for a numerical and a categorical feature. |
| `pd_cat_cat_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the :py:meth:`alibi.utils.visualization.heatmap` functon when plotting the PD for two categorical features. |
| `fig_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.figure.set`_ function. .. _matplotlib.pyplot.plot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html .. _matplotlib.pyplot.contourf: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html .. _matplotlib.figure.set: https://matplotlib.org/stable/api/figure_api.html |

**Returns**
- Type: `np.ndarray`
