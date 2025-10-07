# `alibi.utils.mapping`
## Functions
### `num_to_ord`

```python
num_to_ord(data: numpy.ndarray, dist: dict) -> numpy.ndarray
```

Transform numerical values into categories using the map calculated under the fit method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  |  |
| `dist` | `dict` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `ohe_to_ord`

```python
ohe_to_ord(X_ohe: numpy.ndarray, cat_vars_ohe: dict) -> Tuple[numpy.ndarray, dict]
```

Convert one-hot encoded variables to ordinal encodings.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  |  |
| `cat_vars_ohe` | `dict` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, dict]`

### `ohe_to_ord_shape`

```python
ohe_to_ord_shape(shape: tuple, cat_vars: Dict[int, int], is_ohe: bool = False) -> tuple
```

Infer shape of instance if the categorical variables have ordinal instead of one-hot encoding.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shape` | `tuple` |  |  |
| `cat_vars` | `Dict[int, int]` |  |  |
| `is_ohe` | `bool` | `False` |  |

**Returns**
- Type: `tuple`

### `ord_to_num`

```python
ord_to_num(data: numpy.ndarray, dist: dict) -> numpy.ndarray
```

Transform categorical into numerical values using a mapping.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  |  |
| `dist` | `dict` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `ord_to_ohe`

```python
ord_to_ohe(X_ord: numpy.ndarray, cat_vars_ord: dict) -> Tuple[numpy.ndarray, dict]
```

Convert ordinal to one-hot encoded variables.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ord` | `numpy.ndarray` |  |  |
| `cat_vars_ord` | `dict` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, dict]`
