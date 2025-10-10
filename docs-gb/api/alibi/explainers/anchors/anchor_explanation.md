# `alibi.explainers.anchors.anchor_explanation`
## `AnchorExplanation`

### Constructor

```python
AnchorExplanation(self, exp_type: str, exp_map: dict) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `exp_type` | `str` |  | Type of explainer: tabular, text or image. |
| `exp_map` | `dict` |  | Dictionary with the anchors and explainer metadata for an observation. |

### Methods

#### `coverage`

```python
coverage(partial_index: Optional[int] = None) -> float
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` | Get the result coverage until a certain index. For example, if the result has precisions ``[0.1, 0.5, 0.95]`` and ``partial_index=1``, this will return ``0.5``. |

**Returns**
- Type: `float`

#### `examples`

```python
examples(only_different_prediction: bool = False, only_same_prediction: bool = False, partial_index: Optional[int] = None) -> Union[list, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `only_different_prediction` | `bool` | `False` | If ``True``, will only return examples where the result makes a different prediction than the original model. |
| `only_same_prediction` | `bool` | `False` | If ``True``, will only return examples where the result makes the same prediction than the original model. |
| `partial_index` | `Optional[int]` | `None` | Get the examples from the partial result until a certain index. |

**Returns**
- Type: `Union[list, numpy.ndarray]`

#### `features`

```python
features(partial_index: Optional[int] = None) -> list
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` | Get the result until a certain index. For example, if the result uses ``segment_labels=(1, 2, 3)`` and ``partial_index=1``, this will return ``[1, 2]``. |

**Returns**
- Type: `list`

#### `names`

```python
names(partial_index: Optional[int] = None) -> list
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` | Get the result until a certain index. For example, if the result is ``(A=1, B=2, C=2)`` and ``partial_index=1``, this will return ``["A=1", "B=2"]``. |

**Returns**
- Type: `list`

#### `precision`

```python
precision(partial_index: Optional[int] = None) -> float
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `partial_index` | `Optional[int]` | `None` | Get the result precision until a certain index. For example, if the result has precisions ``[0.1, 0.5, 0.95]`` and ``partial_index=1``, this will return ``0.5``. |

**Returns**
- Type: `float`
