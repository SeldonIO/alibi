# `alibi.confidence.trustscore`
## Constants
### `logger`
```python
logger: Logger = <Logger alibi.confidence.trustscore (WARNING)>
```
## `TrustScore`

### Constructor

```python
TrustScore(self, k_filter: int = 10, alpha: float = 0.0, filter_type: Optional[str] = None, leaf_size: int = 40, metric: str = 'euclidean', dist_filter_type: str = 'point') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k_filter` | `int` | `10` |  |
| `alpha` | `float` | `0.0` |  |
| `filter_type` | `Optional[str]` | `None` |  |
| `leaf_size` | `int` | `40` |  |
| `metric` | `str` | `'euclidean'` |  |
| `dist_filter_type` | `str` | `'point'` |  |

### Methods

#### `filter_by_distance_knn`

```python
filter_by_distance_knn(X: numpy.ndarray) -> numpy.ndarray
```

Filter out instances with low kNN density. Calculate distance to k-nearest point in the data for each

instance and remove instances above a cutoff distance.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `filter_by_probability_knn`

```python
filter_by_probability_knn(X: numpy.ndarray, Y: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Filter out instances with high label disagreement amongst its k nearest neighbors.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `fit`

```python
fit(X: numpy.ndarray, Y: numpy.ndarray, classes: Optional[int] = None) -> None
```

Build KDTrees for each prediction class.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |
| `classes` | `Optional[int]` | `None` |  |

**Returns**
- Type: `None`

#### `score`

```python
score(X: numpy.ndarray, Y: numpy.ndarray, k: int = 2, dist_type: str = 'point') -> Tuple[numpy.ndarray, numpy.ndarray]
```

Calculate trust scores = ratio of distance to closest class other than the

predicted class to distance to predicted class.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y` | `numpy.ndarray` |  |  |
| `k` | `int` | `2` |  |
| `dist_type` | `str` | `'point'` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
