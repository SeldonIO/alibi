# `alibi.explainers.anchors.anchor_text`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `DEFAULT_DATA_ANCHOR`
```python
DEFAULT_DATA_ANCHOR: dict = {'anchor': [], 'precision': None, 'coverage': None, 'raw': None}
```

### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.anchors.anchor_text (WARNING)>
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

### `DEFAULT_SAMPLING_UNKNOWN`
```python
DEFAULT_SAMPLING_UNKNOWN: dict = {'sample_proba': 0.5}
```

### `DEFAULT_SAMPLING_SIMILARITY`
```python
DEFAULT_SAMPLING_SIMILARITY: dict = {'sample_proba': 0.5, 'top_n': 100, 'temperature': 1.0, 'use_proba': False}
```

### `DEFAULT_SAMPLING_LANGUAGE_MODEL`
```python
DEFAULT_SAMPLING_LANGUAGE_MODEL: dict = {'filling': 'parallel', 'sample_proba': 0.5, 'top_n': 100, 'temperature': 1.0...
```

## `AnchorText`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Constructor

```python
AnchorText(self, predictor: Callable[[List[str]], numpy.ndarray], sampling_strategy: str = 'unknown', nlp: Optional[ForwardRef('spacy.language.Language')] = None, language_model: Optional[ForwardRef('LanguageModel')] = None, seed: int = 0, **kwargs: Any) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[typing.List[str]]], numpy.ndarray]` |  |  |
| `sampling_strategy` | `str` | `'unknown'` |  |
| `nlp` | `Optional[spacy.language.Language]` | `None` |  |
| `language_model` | `Optional[alibi.utils.lang_model.LanguageModel]` | `None` |  |
| `seed` | `int` | `0` |  |

### Methods

### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

Compute the agreement between a classifier prediction on an instance to be explained

and the prediction on a set of samples which have a subset of features fixed to a
given value (aka compute the precision of anchors).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `explain`

```python
explain(text: str, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = True, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

Explain instance and return anchor with metadata.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |
| `threshold` | `float` | `0.95` |  |
| `delta` | `float` | `0.1` |  |
| `tau` | `float` | `0.15` |  |
| `batch_size` | `int` | `100` |  |
| `coverage_samples` | `int` | `10000` |  |
| `beam_size` | `int` | `1` |  |
| `stop_on_first` | `bool` | `True` |  |
| `max_anchor_size` | `Optional[int]` | `None` |  |
| `min_samples_start` | `int` | `100` |  |
| `n_covered_ex` | `int` | `10` |  |
| `binary_cache_size` | `int` | `10000` |  |
| `cache_margin` | `int` | `1000` |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

### `sampler`

```python
sampler(anchor: Tuple[int, tuple], num_samples: int, compute_labels: bool = True) -> Union[List[Union[numpy.ndarray, float, int]], List[numpy.ndarray]]
```

Generate perturbed samples while maintaining features in positions specified in

anchor unchanged.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `Tuple[int, tuple]` |  |  |
| `num_samples` | `int` |  |  |
| `compute_labels` | `bool` | `True` |  |

**Returns**
- Type: `Union[List[Union[numpy.ndarray, float, int]], List[numpy.ndarray]]`
