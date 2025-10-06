# `alibi.utils.discretizer`
## Classes
### `Discretizer`

#### Constructor

```python
Discretizer(self, data: numpy.ndarray, numerical_features: List[int], feature_names: List[str], percentiles: Sequence[Union[int, float]] = (25, 50, 75)) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |
| `numerical_features` | `List[int]` |  | List of indices corresponding to the continuous feature columns. Only these features will be discretized. |
| `feature_names` | `List[str]` |  | List with feature names. |
| `percentiles` | `Sequence[Union[int, float]]` | `(25, 50, 75)` | Percentiles used for discretization. |

#### Methods

##### `bins`

```python
bins(data: numpy.ndarray) -> List[numpy.ndarray]
```

Parameters

----------
data
    Data to discretize.

Returns
-------
List with bin values for each feature that is discretized.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |

**Returns**
- Type: `List[numpy.ndarray]`

##### `discretize`

```python
discretize(data: numpy.ndarray) -> numpy.ndarray
```

Parameters

----------
data
    Data to discretize.

Returns
-------
Discretized version of data with the same dimension.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize. |

**Returns**
- Type: `numpy.ndarray`

##### `get_percentiles`

```python
get_percentiles(x: numpy.ndarray, qts: numpy.ndarray) -> numpy.ndarray
```

Discretizes the the data in `x` using the quantiles in `qts`.

This is achieved by searching for the index of each value in `x`
into `qts`, which is assumed to be a 1-D sorted array.

Parameters
----------
x
    A `numpy` array of data to be discretized
qts:
    A `numpy` array of percentiles. This should be a 1-D array sorted in
    ascending order.

Returns
-------
A discretized data `numpy` array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | A `numpy` array of data to be discretized |
| `qts` | `numpy.ndarray` |  | A `numpy` array of percentiles. This should be a 1-D array sorted in ascending order. |

**Returns**
- Type: `numpy.ndarray`
