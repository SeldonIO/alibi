# `alibi.explainers.integrated_gradients`
## `IntegratedGradients`

_Inherits from:_ `Explainer`, `ABC`, `Base`

### Constructor

```python
IntegratedGradients(self, model: keras.src.models.model.Model, layer: Union[Callable[[keras.src.models.model.Model], keras.src.layers.layer.Layer], keras.src.layers.layer.Layer, NoneType] = None, target_fn: Optional[Callable] = None, method: str = 'gausslegendre', n_steps: int = 50, internal_batch_size: int = 100) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  |  |
| `layer` | `Union[Callable[[.[<class 'keras.src.models.model.Model'>]], keras.src.layers.layer.Layer], keras.src.layers.layer.Layer, None]` | `None` |  |
| `target_fn` | `Optional[Callable]` | `None` |  |
| `method` | `str` | `'gausslegendre'` |  |
| `n_steps` | `int` | `50` |  |
| `internal_batch_size` | `int` | `100` |  |

### Methods

#### `explain`

```python
explain(X: Union[numpy.ndarray, List[numpy.ndarray]], forward_kwargs: Optional[dict] = None, baselines: Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None] = None, target: Union[int, list, numpy.ndarray, None] = None, attribute_to_layer_inputs: bool = False) -> alibi.api.interfaces.Explanation
```

Calculates the attributions for each input feature or element of layer and

returns an Explanation object.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, List[numpy.ndarray]]` |  |  |
| `forward_kwargs` | `Optional[dict]` | `None` |  |
| `baselines` | `Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None]` | `None` |  |
| `target` | `Union[int, list, numpy.ndarray, None]` | `None` |  |
| `attribute_to_layer_inputs` | `bool` | `False` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: keras.src.models.model.Model) -> None
```

Resets the predictor model.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `None`

## `LayerState`

_Inherits from:_ `str`, `Enum`

An enumeration.
