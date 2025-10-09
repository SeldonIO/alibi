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

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `Union[pandas.core.frame.DataFrame, numpy.ndarray]` |  |  |
| `categorical_columns` | `Union[List[int], List[str], None]` | `None` |  |

**Returns**
- Type: `Dict[int, list]`
