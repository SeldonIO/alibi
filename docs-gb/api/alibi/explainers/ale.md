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
