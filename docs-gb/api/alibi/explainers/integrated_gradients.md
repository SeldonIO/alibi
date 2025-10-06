# `alibi.explainers.integrated_gradients`
## Constants
### `DEFAULT_DATA_INTGRAD`
```python
DEFAULT_DATA_INTGRAD: dict = {'attributions': None, 'X': None, 'forward_kwargs': None, 'baselines': None, ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_INTGRAD`
```python
DEFAULT_META_INTGRAD: dict = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

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

## Classes
### `IntegratedGradients` (_inherits from `Explainer`, `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

```python
IntegratedGradients(self, model: keras.src.models.model.Model, layer: Union[Callable[[keras.src.models.model.Model], keras.src.layers.layer.Layer], keras.src.layers.layer.Layer, NoneType] = None, target_fn: Optional[Callable] = None, method: str = 'gausslegendre', n_steps: int = 50, internal_batch_size: int = 100) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  | `tensorflow` model. |
| `layer` | `Union[Callable[[.[<class 'keras.src.models.model.Model'>]], keras.src.layers.layer.Layer], keras.src.layers.layer.Layer, None]` | `None` | A layer or a function having as parameter the model and returning a layer with respect to which the gradients are calculated. If not provided, the gradients are calculated with respect to the input. To guarantee saving and loading of the explainer, the layer has to be specified as a callable which returns a layer given the model. E.g. ``lambda model: model.layers[0].embeddings``. |
| `target_fn` | `Optional[Callable]` | `None` | A scalar function that is applied to the predictions of the model. This can be used to specify which scalar output the attributions should be calculated for. This can be particularly useful if the desired output is not known before calling the model (e.g. explaining the `argmax` output for a probabilistic classifier, in this case we could pass ``target_fn=partial(np.argmax, axis=1)``). |
| `method` | `str` | `'gausslegendre'` | Method for the integral approximation. Methods available: ``"riemann_left"``, ``"riemann_right"``, ``"riemann_middle"``, ``"riemann_trapezoid"``, ``"gausslegendre"``. |
| `n_steps` | `int` | `50` | Number of step in the path integral approximation from the baseline to the input instance. |
| `internal_batch_size` | `int` | `100` | Batch size for the internal batching. |

#### Methods

##### `explain`

```python
explain(X: Union[numpy.ndarray, List[numpy.ndarray]], forward_kwargs: Optional[dict] = None, baselines: Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None] = None, target: Union[int, list, numpy.ndarray, None] = None, attribute_to_layer_inputs: bool = False) -> alibi.api.interfaces.Explanation
```

Calculates the attributions for each input feature or element of layer and

returns an Explanation object.

Parameters
----------
X
    Instance for which integrated gradients attribution are computed.
forward_kwargs
    Input keyword args. If it's not ``None``, it must be a dict with `numpy` arrays as values.
    The first dimension of the arrays must correspond to the number of examples.
    It will be repeated for each of `n_steps` along the integrated path.
    The attributions are not computed with respect to these arguments.
baselines
    Baselines (starting point of the path integral) for each instance.
    If the passed value is an `np.ndarray` must have the same shape as `X`.
    If not provided, all features values for the baselines are set to 0.
target
    Defines which element of the model output is considered to compute the gradients.
    Target can be a numpy array, a list or a numeric value.
    Numeric values are only valid if the model's output is a rank-n tensor
    with n <= 2 (regression and classification models).
    If a numeric value is passed, the gradients are calculated for
    the same element of the output for all data points.
    For regression models whose output is a scalar, target should not be provided.
    For classification models `target` can be either the true classes or the classes predicted by the model.
    It must be provided for classification models and regression models whose output is a vector.
    If the model's output is a rank-n tensor with n > 2,
    the target must be a rank-2 numpy array or a list of lists (a matrix) with dimensions nb_samples X (n-1) .
attribute_to_layer_inputs
    In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
    outputs. If ``True``, gradients are computed for the layer's inputs, if ``False`` for the layer's outputs.

Returns
-------
explanation
    `Explanation` object including `meta` and `data` attributes with integrated gradients attributions
    for each feature. See usage at `IG examples`_ for details.

    .. _IG examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, List[numpy.ndarray]]` |  | Instance for which integrated gradients attribution are computed. |
| `forward_kwargs` | `Optional[dict]` | `None` | Input keyword args. If it's not ``None``, it must be a dict with `numpy` arrays as values. The first dimension of the arrays must correspond to the number of examples. It will be repeated for each of `n_steps` along the integrated path. The attributions are not computed with respect to these arguments. |
| `baselines` | `Union[int, float, numpy.ndarray, List[int], List[float], List[numpy.ndarray], None]` | `None` | Baselines (starting point of the path integral) for each instance. If the passed value is an `np.ndarray` must have the same shape as `X`. If not provided, all features values for the baselines are set to 0. |
| `target` | `Union[int, list, numpy.ndarray, None]` | `None` | Defines which element of the model output is considered to compute the gradients. Target can be a numpy array, a list or a numeric value. Numeric values are only valid if the model's output is a rank-n tensor with n <= 2 (regression and classification models). If a numeric value is passed, the gradients are calculated for the same element of the output for all data points. For regression models whose output is a scalar, target should not be provided. For classification models `target` can be either the true classes or the classes predicted by the model. It must be provided for classification models and regression models whose output is a vector. If the model's output is a rank-n tensor with n > 2, the target must be a rank-2 numpy array or a list of lists (a matrix) with dimensions nb_samples X (n-1) . |
| `attribute_to_layer_inputs` | `bool` | `False` | In case of layers gradients, controls whether the gradients are computed for the layer's inputs or outputs. If ``True``, gradients are computed for the layer's inputs, if ``False`` for the layer's outputs. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `reset_predictor`

```python
reset_predictor(predictor: keras.src.models.model.Model) -> None
```

Resets the predictor model.

Parameters
----------
predictor
    New prediction model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `keras.src.models.model.Model` |  | New prediction model. |

**Returns**
- Type: `None`

### `LayerState` (_inherits from `str`, `Enum`)

An enumeration.

#### Constructor

```python
LayerState(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |
