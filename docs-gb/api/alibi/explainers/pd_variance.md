# `alibi.explainers.pd_variance`
## Constants
### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD: dict = {'feature_deciles': None, 'pd_values': None, 'ice_values': None, 'feature_val...
```

### `DEFAULT_DATA_PDVARIANCE`
```python
DEFAULT_DATA_PDVARIANCE: dict = {'feature_deciles': None, 'pd_values': None, 'feature_values': None, 'feature...
```

### `DEFAULT_META_PDVARIANCE`
```python
DEFAULT_META_PDVARIANCE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.pd_variance (WARNING)>
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

## `Method`

_Inherits from:_ `str`, `Enum`

Enumeration of supported methods.

## `PartialDependenceVariance`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Implementation of the partial dependence(PD) variance feature importance and feature interaction for
tabular datasets. The method measure the importance feature importance as the variance within the PD function.
Similar, the potential feature interaction is measured by computing the variance within the two-way PD function
by holding one variable constant and letting the other vary. Supports black-box models and the following `sklearn`
tree-based models: `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`,
`HistGradientBoostingRegressor`, `HistGradientBoostingRegressor`, `DecisionTreeRegressor`,
`RandomForestRegressor`.

For details of the method see the original paper: https://arxiv.org/abs/1805.04755 .

### Constructor

```python
PartialDependenceVariance(self, predictor: Union[sklearn.base.BaseEstimator, Callable[[numpy.ndarray], numpy.ndarray]], feature_names: Optional[List[str]] = None, categorical_names: Optional[Dict[int, List[str]]] = None, target_names: Optional[List[str]] = None, verbose: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[sklearn.base.BaseEstimator, Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` |  | A `sklearn` estimator or a prediction function which receives as input a `numpy` array of size `N x F` and outputs a `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances, `F` is the number of features and `T` is the number of targets. |
| `feature_names` | `Optional[List[str]]` | `None` | A list of feature names used for displaying results.E |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. Necessary to identify the categorical features in the dataset. An example for `categorical_names` would be:: category_map = {0: ["married", "divorced"], 3: ["high school diploma", "master's degree"]} |
| `target_names` | `Optional[List[str]]` | `None` | A list of target/output names used for displaying results. |
| `verbose` | `bool` | `False` | Whether to print the progress of the explainer. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Union[List[int], List[Tuple[int, int]], None] = None, method: Literal[importance, interaction] = 'importance', percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | A `N x F` tabular dataset used to calculate partial dependence curves. This is typically the training dataset or a representative sample. |
| `features` | `Union[List[int], List[Tuple[int, int]], None]` | `None` | A list of features for which to compute the feature importance or a list of feature pairs for which to compute the feature interaction. Some example of `features` would be: ``[0, 1, 3]``, ``[(0, 1), (0, 3), (1, 3)]``, where ``0``,``1``, and ``3`` correspond to the columns 0, 1, and 3 in `X`. If not provided, the feature importance or the feature interaction will be computed for every feature or for every combination of feature pairs, depending on the parameter `method`. |
| `method` | `Literal[importance, interaction]` | `'importance'` | Flag to specify whether to compute the feature importance or the feature interaction of the elements provided in `features`. Supported values: ``'importance'`` | ``'interaction'``. |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` | Lower and upper percentiles used to limit the feature values to potentially remove outliers from low-density regions. Note that for features with not many data points with large/low values, the PD estimates are less reliable in those extreme regions. The values must be in [0, 1]. Only used with `grid_resolution`. |
| `grid_resolution` | `int` | `100` | Number of equidistant points to split the range of each target feature. Only applies if the number of unique values of a target feature in the reference dataset `X` is greater than the `grid_resolution` value. For example, consider a case where a feature can take the following values: ``[0.1, 0.3, 0.35, 0.351, 0.4, 0.41, 0.44, ..., 0.5, 0.54, 0.56, 0.6, 0.65, 0.7, 0.9]``, and we are not interested in evaluating the marginal effect at every single point as it can become computationally costly (assume hundreds/thousands of points) without providing any additional information for nearby points (e.g., 0.35 and 351). By setting ``grid_resolution=5``, the marginal effect is computed for the values ``[0.1, 0.3, 0.5, 0.7, 0.9]`` instead, which is less computationally demanding and can provide similar insights regarding the model's behaviour. Note that the extreme values of the grid can be controlled using the `percentiles` argument. |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` | Custom grid points. Must be a `dict` where the keys are the target features indices and the values are monotonically increasing arrays defining the grid points for a numerical feature, and a subset of categorical feature values for a categorical feature. If the `grid_points` are not specified, then the grid will be constructed based on the unique target feature values available in the dataset `X`, or based on the `grid_resolution` and `percentiles` (check `grid_resolution` to see when it applies). For categorical features, the corresponding value in the `grid_points` can be specified either as array of strings or array of integers corresponding the label encodings. Note that the label encoding must match the ordering of the values provided in the `categorical_names`. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `plot_pd_variance`

```python
plot_pd_variance(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', targets: Union[List[Union[str, int]], Literal[all]] = 'all', summarise: bool = True, n_cols: int = 3, sort: bool = True, top_k: Optional[int] = None, plot_limits: Optional[Tuple[float, float]] = None, ax: Union[matplotlib.axes._axes.Axes, numpy.ndarray, None] = None, sharey: Optional[Literal[all, row]] = 'all', bar_kw: Optional[dict] = None, line_kw: Optional[dict] = None, fig_kw: Optional[dict] = None)
```

Plot feature importance and feature interaction based on partial dependence curves on `matplotlib` axes.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  | An `Explanation` object produced by a call to the :py:meth:`alibi.explainers.pd_variance.PartialDependenceVariance.explain` method. |
| `features` | `Union[List[int], Literal[all]]` | `'all'` | A list of features entries provided in `feature_names` argument  to the :py:meth:`alibi.explainers.pd_variance.PartialDependenceVariance.explain` method, or ``'all'`` to  plot all the explained features. For example, if  ``feature_names = ['temp', 'hum', 'windspeed']`` and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set ``features=[0, 2]``. Defaults to ``'all'``. |
| `targets` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` | A target name/index, or a list of target names/indices, for which to plot the feature importance/interaction, or ``'all'``. Can be a mix of integers denoting target index or strings denoting entries in `exp.meta['params']['target_names']`. By default ``'all'`` to plot the importance for all features or to plot all the feature interactions. |
| `summarise` | `bool` | `True` | Whether to plot only the summary of the feature importance/interaction as a bar plot, or plot comprehensive exposition including partial dependence plots and conditional importance plots. |
| `n_cols` | `int` | `3` | Number of columns to organize the resulting plot into. |
| `sort` | `bool` | `True` | Boolean flag whether to sort the values in descending order. |
| `top_k` | `Optional[int]` | `None` | Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed. |
| `plot_limits` | `Optional[Tuple[float, float]]` | `None` | Minimum and maximum y-limits for all the line plots. If ``None`` will be automatically inferred. |
| `ax` | `Union[matplotlib.axes._axes.Axes, numpy.ndarray, None]` | `None` | A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on. |
| `sharey` | `Optional[Literal[all, row]]` | `'all'` | A parameter specifying whether the y-axis of the PD and ICE curves should be on the same scale for several features. Possible values are: ``'all'`` | ``'row'`` | ``None``. |
| `bar_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.barh`_ function. |
| `line_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.pyplot.plot`_ function. |
| `fig_kw` | `Optional[dict]` | `None` | Keyword arguments passed to the `matplotlib.figure.set`_ function. .. _matplotlib.pyplot.barh: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html .. _matplotlib.pyplot.plot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html .. _matplotlib.figure.set: https://matplotlib.org/stable/api/figure_api.html |
