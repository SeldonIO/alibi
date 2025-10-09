# `alibi.explainers.partial_dependence`
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
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `target_names` | `Optional[List[str]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

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
| `predictor` | `Union[sklearn.base.BaseEstimator, Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]` |  |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `target_names` | `Optional[List[str]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, int]]]] = None, kind: Literal[average, individual, both] = 'average', percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

Calculates the partial dependence for each feature and/or tuples of features with respect to the all targets

and the reference dataset `X`.

Parameters

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

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], sklearn.base.BaseEstimator]) -> None
```

Resets the predictor function or tree-based `sklearn` estimator.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], sklearn.base.BaseEstimator]` |  |  |

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
| `predictor` | `sklearn.base.BaseEstimator` |  |  |
| `feature_names` | `Optional[List[str]]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `target_names` | `Optional[List[str]]` | `None` |  |
| `verbose` | `bool` | `False` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, features: Optional[List[Union[int, Tuple[int, int]]]] = None, percentiles: Tuple[float, float] = (0.0, 1.0), grid_resolution: int = 100, grid_points: Optional[Dict[int, Union[List[Any], numpy.ndarray]]] = None) -> alibi.api.interfaces.Explanation
```

Calculates the partial dependence for each feature and/or tuples of features with respect to the all targets

and the reference dataset `X`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `features` | `Optional[List[Union[int, Tuple[int, int]]]]` | `None` |  |
| `percentiles` | `Tuple[float, float]` | `(0.0, 1.0)` |  |
| `grid_resolution` | `int` | `100` |  |
| `grid_points` | `Optional[Dict[int, Union[List[Any], numpy.ndarray]]]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `plot_pd`

```python
plot_pd(exp: alibi.api.interfaces.Explanation, features: Union[List[int], Literal[all]] = 'all', target: Union[str, int] = 0, n_cols: int = 3, n_ice: Union[Literal[all], int, List[int]] = 100, center: bool = False, pd_limits: Optional[Tuple[float, float]] = None, levels: int = 8, ax: Union[ForwardRef('plt.Axes'), numpy.ndarray, None] = None, sharey: Optional[Literal[all, row]] = 'all', pd_num_kw: Optional[dict] = None, ice_num_kw: Optional[dict] = None, pd_cat_kw: Optional[dict] = None, ice_cat_kw: Optional[dict] = None, pd_num_num_kw: Optional[dict] = None, pd_num_cat_kw: Optional[dict] = None, pd_cat_cat_kw: Optional[dict] = None, fig_kw: Optional[dict] = None) -> np.ndarray
```

Plot partial dependence curves on matplotlib axes.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp` | `alibi.api.interfaces.Explanation` |  |  |
| `features` | `Union[List[int], Literal[all]]` | `'all'` |  |
| `target` | `Union[str, int]` | `0` |  |
| `n_cols` | `int` | `3` |  |
| `n_ice` | `Union[Literal[all], int, List[int]]` | `100` |  |
| `center` | `bool` | `False` |  |
| `pd_limits` | `Optional[Tuple[float, float]]` | `None` |  |
| `levels` | `int` | `8` |  |
| `ax` | `Union[ForwardRef('plt.Axes'), numpy.ndarray, None]` | `None` |  |
| `sharey` | `Optional[Literal[all, row]]` | `'all'` |  |
| `pd_num_kw` | `Optional[dict]` | `None` |  |
| `ice_num_kw` | `Optional[dict]` | `None` |  |
| `pd_cat_kw` | `Optional[dict]` | `None` |  |
| `ice_cat_kw` | `Optional[dict]` | `None` |  |
| `pd_num_num_kw` | `Optional[dict]` | `None` |  |
| `pd_num_cat_kw` | `Optional[dict]` | `None` |  |
| `pd_cat_cat_kw` | `Optional[dict]` | `None` |  |
| `fig_kw` | `Optional[dict]` | `None` |  |

**Returns**
- Type: `np.ndarray`
