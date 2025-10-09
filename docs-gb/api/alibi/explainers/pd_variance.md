# `alibi.explainers.pd_variance`
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
