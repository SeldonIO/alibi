# `alibi.utils.discretizer`
## `Discretizer`

### Constructor

```python
Discretizer(self, data: numpy.ndarray, numerical_features: List[int], feature_names: List[str], percentiles: Sequence[Union[int, float]] = (25, 50, 75)) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |
| `numerical_features` | `List[int]` |  | List of indices corresponding to the continuous feature columns. Only these features will be discretized. |
| `feature_names` | `List[str]` |  | List with feature names. |
| `percentiles` | `Sequence[Union[int, float]]` | `(25, 50, 75)` | Percentiles used for discretization. |

### Methods

#### `bins`

```python
bins(data: numpy.ndarray) -> List[numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |

**Returns**
- Type: `List[numpy.ndarray]`

#### `discretize`

```python
discretize(data: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |

**Returns**
- Type: `numpy.ndarray`

#### `get_percentiles`

```python
get_percentiles(x: numpy.ndarray, qts: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | A `numpy` array of data to be discretized |
| `qts` | `numpy.ndarray` |  | A `numpy` array of percentiles. This should be a 1-D array sorted in ascending order. |

**Returns**
- Type: `numpy.ndarray`
