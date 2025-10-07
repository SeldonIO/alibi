# `alibi.explainers.integrated_gradients`
## Constants
### `DEFAULT_DATA_INTGRAD`
```python
DEFAULT_DATA_INTGRAD: dict = {'attributions': None, 'X': None, 'forward_kwargs': None, 'baselines': None, ...
```

### `DEFAULT_META_INTGRAD`
```python
DEFAULT_META_INTGRAD: dict = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.integrated_gradients (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

## `IntegratedGradients`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

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

### `explain`

```python
explain(X: Union[numpy.ndarray, List[numpy.ndarray]], forward_kwargs: Optional[dict] = None, baselines: Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None] = None, target: Union[int, list, numpy.ndarray, None] = None, attribute_to_layer_inputs: bool = False) -> alibi.api.interfaces.Explanation
```

Calculates the attributions for each input feature or element of layer and

returns an Explanation object.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, List[numpy.ndarray]]` |  |  |
| `forward_kwargs` | `Optional[dict]` | `None` |  |
| `baselines` | `Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None]` | `None` |  |
| `target` | `Union[int, list, numpy.ndarray, None]` | `None` |  |
| `attribute_to_layer_inputs` | `bool` | `False` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

### `reset_predictor`

```python
reset_predictor(predictor: keras.src.models.model.Model) -> None
```

Resets the predictor model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `None`

## `LayerState`

_Inherits from:_ `str`, `Enum`

An enumeration.

### Constructor

```python
LayerState(self, /, *args, **kwargs)
```
