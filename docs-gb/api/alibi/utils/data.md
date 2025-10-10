# `alibi.utils.data`
## `Bunch`

_Inherits from:_ `dict`

Container object for internal datasets.
Dictionary-like object that exposes its keys as attributes.

### Constructor

```python
Bunch(self, **kwargs)
```

## Functions
### `gen_category_map`

```python
gen_category_map(data: Union[pandas.core.frame.DataFrame, numpy.ndarray], categorical_columns: Union[List[int], List[str], None] = None) -> Dict[int, list]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `Union[pandas.core.frame.DataFrame, numpy.ndarray]` |  | 2-dimensional `pandas` dataframe or `numpy` array. |
| `categorical_columns` | `Union[List[int], List[str], None]` | `None` | A list of columns indicating categorical variables. Optional if passing a `pandas` dataframe as inference will be used based on dtype ``'O'``. If passing a `numpy` array this is compulsory. |

**Returns**
- Type: `Dict[int, list]`
