# `alibi.utils.mapping`
## Functions
### `num_to_ord`

```python
num_to_ord(data: numpy.ndarray, dist: dict) -> numpy.ndarray
```

Transform numerical values into categories using the map calculated under the fit method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | `Numpy` array with the numerical data. |
| `dist` | `dict` |  | Dict with as keys the categorical variables and as values the numerical value for each category. |

**Returns**
- Type: `numpy.ndarray`

### `ohe_to_ord`

```python
ohe_to_ord(X_ohe: numpy.ndarray, cat_vars_ohe: dict) -> Tuple[numpy.ndarray, dict]
```

Convert one-hot encoded variables to ordinal encodings.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  | Data with mixture of one-hot encoded and numerical variables. |
| `cat_vars_ohe` | `dict` |  | Dict with as keys the first column index for each one-hot encoded categorical variable and as values the number of categories per categorical variable. |

**Returns**
- Type: `Tuple[numpy.ndarray, dict]`

### `ohe_to_ord_shape`

```python
ohe_to_ord_shape(shape: tuple, cat_vars: Dict[int, int], is_ohe: bool = False) -> tuple
```

Infer shape of instance if the categorical variables have ordinal instead of one-hot encoding.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shape` | `tuple` |  | Instance shape, starting with batch dimension. |
| `cat_vars` | `Dict[int, int]` |  | Dict with as keys the categorical columns and as values the number of categories per categorical variable. |
| `is_ohe` | `bool` | `False` | Whether instance is OHE. |

**Returns**
- Type: `tuple`

### `ord_to_num`

```python
ord_to_num(data: numpy.ndarray, dist: dict) -> numpy.ndarray
```

Transform categorical into numerical values using a mapping.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | `Numpy` array with the categorical data. |
| `dist` | `dict` |  | Dict with as keys the categorical variables and as values the numerical value for each category. |

**Returns**
- Type: `numpy.ndarray`

### `ord_to_ohe`

```python
ord_to_ohe(X_ord: numpy.ndarray, cat_vars_ord: dict) -> Tuple[numpy.ndarray, dict]
```

Convert ordinal to one-hot encoded variables.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ord` | `numpy.ndarray` |  | Data with mixture of ordinal encoded and numerical variables. |
| `cat_vars_ord` | `dict` |  | Dict with as keys the categorical columns and as values the number of categories per categorical variable. |

**Returns**
- Type: `Tuple[numpy.ndarray, dict]`
