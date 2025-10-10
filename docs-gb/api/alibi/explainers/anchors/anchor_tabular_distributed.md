# `alibi.explainers.anchors.anchor_tabular_distributed`
## `DistributedAnchorBaseBeam`

_Inherits from:_ `AnchorBaseBeam`

### Constructor

```python
DistributedAnchorBaseBeam(self, samplers: List[Callable], **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samplers` | `List[Callable]` |  | Objects that can be called with args (`result`, `n_samples`) tuple to draw samples. |

### Methods

#### `draw_samples`

```python
draw_samples(anchors: list, batch_size: int) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  |  |
| `batch_size` | `int` |  |  |
| `anchors,` | `batch_size` |  | See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.draw_samples` implementation. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

## `DistributedAnchorTabular`

_Inherits from:_ `AnchorTabular`, `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
DistributedAnchorTabular(self, predictor: Callable, feature_names: List[str], categorical_names: Optional[Dict[int, List[str]]] = None, dtype: Type[numpy.generic] = <class 'numpy.float32'>, ohe: bool = False, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. |
| `feature_names` | `List[str]` |  | List with feature names. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` | A `numpy` scalar type that corresponds to the type of input array expected by `predictor`. This may be used to construct arrays of the given type to be passed through the `predictor`. For most use cases this argument should have no effect, but it is exposed for use with predictors that would break when called with an array of unsupported type. |
| `ohe` | `bool` | `False` | Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are assumed to have ordinal encodings. |
| `seed` | `Optional[int]` | `None` | Used to set the random number generator for repeatability purposes. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 1, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `threshold` | `float` | `0.95` |  |
| `delta` | `float` | `0.1` |  |
| `tau` | `float` | `0.15` |  |
| `batch_size` | `int` | `100` |  |
| `coverage_samples` | `int` | `10000` |  |
| `beam_size` | `int` | `1` |  |
| `stop_on_first` | `bool` | `False` |  |
| `max_anchor_size` | `Optional[int]` | `None` |  |
| `min_samples_start` | `int` | `1` |  |
| `n_covered_ex` | `int` | `10` |  |
| `binary_cache_size` | `int` | `10000` |  |
| `cache_margin` | `int` | `1000` |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |
| `X,` | `threshold, delta, tau, batch_size, coverage_samples, beam_size, stop_on_first, max_anchor_size,         min_samples_start, n_covered_ex, binary_cache_size, cache_margin, verbose, verbose_every, **kwargs` |  | See :py:meth:`alibi.explainers.anchors.anchor_tabular.AnchorTabular.explain`. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, disc_perc: tuple = (25, 50, 75), kwargs) -> alibi.explainers.anchors.anchor_tabular.AnchorTabular
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  |  |
| `disc_perc` | `tuple` | `(25, 50, 75)` |  |
| `train_data,` | `disc_perc, **kwargs` |  | See :py:meth:`alibi.explainers.anchors.anchor_tabular.AnchorTabular.fit` superclass. |

**Returns**
- Type: `alibi.explainers.anchors.anchor_tabular.AnchorTabular`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New model prediction function. |

**Returns**
- Type: `None`

## `RemoteSampler`

A wrapper that facilitates the use of `TabularSampler` for distributed sampling.

### Constructor

```python
RemoteSampler(self, *args)
```
### Methods

#### `build_lookups`

```python
build_lookups(X: numpy.ndarray)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | See :py:meth:`alibi.explainers.anchors.anchor_tabular.TabularSampler.build_lookups`. |

#### `set_instance_label`

```python
set_instance_label(X: numpy.ndarray) -> int
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | The instance to be explained. |

**Returns**
- Type: `int`

#### `set_n_covered`

```python
set_n_covered(n_covered: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_covered` | `int` |  | Number of examples where the result (and partial anchors) apply. |

**Returns**
- Type: `None`
