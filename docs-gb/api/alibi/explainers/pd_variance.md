# `alibi.explainers.pd_variance`
## Constants
### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD: dict = { 'feature_deciles': None,
  'feature_names': None,
  'feature_values': None,
  'ice_values': None,
  'pd_values': None}
```
### `DEFAULT_DATA_PDVARIANCE`
```python
DEFAULT_DATA_PDVARIANCE: dict = { 'conditional_importance': None,
  'conditional_importance_values': None,
  'feature_deciles': None,
  'feature_importance': None,
  'feature_interaction': None,
  'feature_names': None,
  'feature_values': None,
  'pd_values': None}
```
### `DEFAULT_META_PDVARIANCE`
```python
DEFAULT_META_PDVARIANCE: dict = {'explanations': ['global'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
### `logger`
```python
logger: Logger = <Logger alibi.explainers.pd_variance (WARNING)>
```
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
| `predictor` | `Union[sklearn.base.BaseEstimator, Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` |  |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `target_names` | `Optional[List[str]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Union[List[int], List[Tuple[int, int]], None] = None, method: Literal[importance, interaction] = 'importance', percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

Calculates the variance partial dependence feature importance for each feature with respect to the all targets

and the reference dataset `X`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `features` | `Union[List[int], List[Tuple[int, int]], None]` | `None` |  |
| `method` | `Literal[importance, interaction]` | `'importance'` |  |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` |  |
| `grid_resolution` | `int` | `100` |  |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `plot_pd_variance`

```python
plot_pd_variance(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', targets: Union[List[Union[str, int]], Literal[all]] = 'all', summarise: bool = True, n_cols: int = 3, sort: bool = True, top_k: Optional[int] = None, plot_limits: Optional[Tuple[float, float]] = None, ax: Union[matplotlib.axes._axes.Axes, numpy.ndarray, None] = None, sharey: Optional[Literal[all, row]] = 'all', bar_kw: Optional[dict] = None, line_kw: Optional[dict] = None, fig_kw: Optional[dict] = None)
```

Plot feature importance and feature interaction based on partial dependence curves on `matplotlib` axes.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  |  |
| `features` | `Union[List[int], Literal[all]]` | `'all'` |  |
| `targets` | `Union[List[Union[str, int]], Literal[all]]` | `'all'` |  |
| `summarise` | `bool` | `True` |  |
| `n_cols` | `int` | `3` |  |
| `sort` | `bool` | `True` |  |
| `top_k` | `Optional[int]` | `None` |  |
| `plot_limits` | `Optional[Tuple[float, float]]` | `None` |  |
| `ax` | `Union[matplotlib.axes._axes.Axes, numpy.ndarray, None]` | `None` |  |
| `sharey` | `Optional[Literal[all, row]]` | `'all'` |  |
| `bar_kw` | `Optional[dict]` | `None` |  |
| `line_kw` | `Optional[dict]` | `None` |  |
| `fig_kw` | `Optional[dict]` | `None` |  |
