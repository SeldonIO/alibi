# `alibi.confidence.trustscore`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.confidence.trustscore (WARNING)>
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

## `TrustScore`

### Constructor

```python
TrustScore(self, k_filter: int = 10, alpha: float = 0.0, filter_type: Optional[str] = None, leaf_size: int = 40, metric: str = 'euclidean', dist_filter_type: str = 'point') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k_filter` | `int` | `10` | Number of neighbors used during either kNN distance or probability filtering. |
| `alpha` | `float` | `0.0` | Fraction of instances to filter out to reduce impact of outliers. |
| `filter_type` | `Optional[str]` | `None` | Filter method: ``'distance_knn'`` | ``'probability_knn'``. |
| `leaf_size` | `int` | `40` | Number of points at which to switch to brute-force. Affects speed and memory required to build trees. Memory to store the tree scales with `n_samples / leaf_size`. |
| `metric` | `str` | `'euclidean'` | Distance metric used for the tree. See `sklearn` DistanceMetric class for a list of available metrics. |
| `dist_filter_type` | `str` | `'point'` | Use either the distance to the k-nearest point (``dist_filter_type = 'point'``) or the average distance from the first to the k-nearest point in the data (``dist_filter_type = 'mean'``). |

### Methods

#### `filter_by_distance_knn`

```python
filter_by_distance_knn(X: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |

**Returns**
- Type: `numpy.ndarray`

#### `filter_by_probability_knn`

```python
filter_by_probability_knn(X: numpy.ndarray, Y: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |
| `Y` | `numpy.ndarray` |  | Predicted class labels. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `fit`

```python
fit(X: numpy.ndarray, Y: numpy.ndarray, classes: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |
| `Y` | `numpy.ndarray` |  | Target labels, either one-hot encoded or the actual class label. |
| `classes` | `Optional[int]` | `None` | Number of prediction classes, needs to be provided if `Y` equals the predicted class. |

**Returns**
- Type: `None`

#### `score`

```python
score(X: numpy.ndarray, Y: numpy.ndarray, k: int = 2, dist_type: str = 'point') -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances to calculate trust score for. |
| `Y` | `numpy.ndarray` |  | Either prediction probabilities for each class or the predicted class. |
| `k` | `int` | `2` | Number of nearest neighbors used for distance calculation. |
| `dist_type` | `str` | `'point'` | Use either the distance to the k-nearest point (``dist_type = 'point'``) or the average distance from the first to the k-nearest point in the data (``dist_type = 'mean'``). |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
