# `alibi.confidence.trustscore`
## Classes
### `TrustScore`

#### Constructor

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

#### Methods

##### `filter_by_distance_knn`

```python
filter_by_distance_knn(X: numpy.ndarray) -> numpy.ndarray
```

Filter out instances with low kNN density. Calculate distance to k-nearest point in the data for each

instance and remove instances above a cutoff distance.

Parameters
----------
X
    Data.

Returns
-------
Filtered data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |

**Returns**
- Type: `numpy.ndarray`

##### `filter_by_probability_knn`

```python
filter_by_probability_knn(X: numpy.ndarray, Y: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Filter out instances with high label disagreement amongst its k nearest neighbors.

Parameters
----------
X
    Data.
Y
    Predicted class labels.

Returns
-------
Filtered data and labels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |
| `Y` | `numpy.ndarray` |  | Predicted class labels. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

##### `fit`

```python
fit(X: numpy.ndarray, Y: numpy.ndarray, classes: Optional[int] = None) -> None
```

Build KDTrees for each prediction class.

Parameters
----------
X
    Data.
Y
    Target labels, either one-hot encoded or the actual class label.
classes
    Number of prediction classes, needs to be provided if `Y` equals the predicted class.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data. |
| `Y` | `numpy.ndarray` |  | Target labels, either one-hot encoded or the actual class label. |
| `classes` | `Optional[int]` | `None` | Number of prediction classes, needs to be provided if `Y` equals the predicted class. |

**Returns**
- Type: `None`

##### `score`

```python
score(X: numpy.ndarray, Y: numpy.ndarray, k: int = 2, dist_type: str = 'point') -> Tuple[numpy.ndarray, numpy.ndarray]
```

Calculate trust scores = ratio of distance to closest class other than the

predicted class to distance to predicted class.

Parameters
----------
X
    Instances to calculate trust score for.
Y
    Either prediction probabilities for each class or the predicted class.
k
    Number of nearest neighbors used for distance calculation.
dist_type
    Use either the distance to the k-nearest point (``dist_type = 'point'``) or
    the average distance from the first to the k-nearest point in the data (``dist_type = 'mean'``).

Returns
-------
Batch with trust scores and the closest not predicted class.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances to calculate trust score for. |
| `Y` | `numpy.ndarray` |  | Either prediction probabilities for each class or the predicted class. |
| `k` | `int` | `2` | Number of nearest neighbors used for distance calculation. |
| `dist_type` | `str` | `'point'` | Use either the distance to the k-nearest point (``dist_type = 'point'``) or the average distance from the first to the k-nearest point in the data (``dist_type = 'mean'``). |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
