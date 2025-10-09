# `alibi.explainers.anchors.anchor_explanation`
## `AnchorExplanation`

### Constructor

```python
AnchorExplanation(self, exp_type: str, exp_map: dict) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp_type` | `str` |  |  |
| `exp_map` | `dict` |  |  |

### Methods

#### `coverage`

```python
coverage(partial_index: Optional[int] = None) -> float
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` |  |

**Returns**
- Type: `float`

#### `examples`

```python
examples(only_different_prediction: bool = False, only_same_prediction: bool = False, partial_index: Optional[int] = None) -> Union[list, numpy.ndarray]
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `only_different_prediction` | `bool` | `False` |  |
| `only_same_prediction` | `bool` | `False` |  |
| `partial_index` | `Optional[int]` | `None` |  |

**Returns**
- Type: `Union[list, numpy.ndarray]`

#### `features`

```python
features(partial_index: Optional[int] = None) -> list
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` |  |

**Returns**
- Type: `list`

#### `names`

```python
names(partial_index: Optional[int] = None) -> list
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` |  |

**Returns**
- Type: `list`

#### `precision`

```python
precision(partial_index: Optional[int] = None) -> float
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` |  |

**Returns**
- Type: `float`
