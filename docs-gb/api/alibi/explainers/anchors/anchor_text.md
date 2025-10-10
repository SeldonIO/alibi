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

### Constructor

```python
AnchorText(self, predictor: Callable[[List[str]], numpy.ndarray], sampling_strategy: str = 'unknown', nlp: Optional[ForwardRef('spacy.language.Language')] = None, language_model: Optional[ForwardRef('LanguageModel')] = None, seed: int = 0, **kwargs: Any) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[typing.List[str]]], numpy.ndarray]` |  | A callable that takes a list of text strings representing `N` data points as inputs and returns `N` outputs. |
| `sampling_strategy` | `str` | `'unknown'` | Perturbation distribution method: - ``'unknown'`` - replaces words with UNKs. - ``'similarity'`` - samples according to a similarity score with the corpus embeddings. - ``'language_model'`` - samples according the language model's output distributions. |
| `nlp` | `Optional[spacy.language.Language]` | `None` | `spaCy` object when sampling method is ``'unknown'`` or ``'similarity'``. |
| `language_model` | `Optional[alibi.utils.lang_model.LanguageModel]` | `None` | Transformers masked language model. This is a model that it adheres to the `LanguageModel` interface we define in :py:class:`alibi.utils.lang_model.LanguageModel`. |
| `seed` | `int` | `0` | If set, ensure identical random streams. |
| `kwargs` | `typing.Any` |  | Sampling arguments can be passed as `kwargs` depending on the `sampling_strategy`. Check default arguments defined in: - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_UNKNOWN` - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_SIMILARITY` - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_LANGUAGE_MODEL` |

### Methods

#### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  | Samples whose labels are to be compared with the instance label. |

**Returns**
- Type: `numpy.ndarray`

#### `explain`

```python
explain(text: str, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = True, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  | Text instance to be explained. |
| `threshold` | `float` | `0.95` | Minimum anchor precision threshold. The algorithm tries to find an anchor that maximizes the coverage under precision constraint. The precision constraint is formally defined as :math:`P(prec(A) \ge t) \ge 1 - \delta`, where :math:`A` is an anchor, :math:`t` is the `threshold` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, we are seeking for an anchor having its precision greater or equal than the given `threshold` with a confidence of `(1 - delta)`. A higher value guarantees that the anchors are faithful to the model, but also leads to more computation time. Note that there are cases in which the precision constraint cannot be satisfied due to the quantile-based discretisation of the numerical features. If that is the case, the best (i.e. highest coverage) non-eligible anchor is returned. |
| `delta` | `float` | `0.1` | Significance threshold. `1 - delta` represents the confidence threshold for the anchor precision (see `threshold`) and the selection of the best anchor candidate in each iteration (see `tau`). |
| `tau` | `float` | `0.15` | Multi-armed bandit parameter used to select candidate anchors in each iteration. The multi-armed bandit algorithm tries to find within a tolerance `tau` the most promising (i.e. according to the precision) `beam_size` candidate anchor(s) from a list of proposed anchors. Formally, when the `beam_size=1`, the multi-armed bandit algorithm seeks to find an anchor :math:`A` such that :math:`P(prec(A) \ge prec(A^\star) - \tau) \ge 1 - \delta`, where :math:`A^\star` is the anchor with the highest true precision (which we don't know), :math:`\tau` is the `tau` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, in each iteration, the algorithm returns with a probability of at least `1 - delta` an anchor :math:`A` with a precision within an error tolerance of `tau` from the precision of the highest true precision anchor :math:`A^\star`. A bigger value for `tau` means faster convergence but also looser anchor conditions. |
| `batch_size` | `int` | `100` | Batch size used for sampling. The Anchor algorithm will query the black-box model in batches of size `batch_size`. A larger `batch_size` gives more confidence in the anchor, again at the expense of computation time since it involves more model prediction calls. |
| `coverage_samples` | `int` | `10000` | Number of samples used to estimate coverage from during anchor search. |
| `beam_size` | `int` | `1` | Number of candidate anchors selected by the multi-armed bandit algorithm in each iteration from a list of proposed anchors. A bigger beam  width can lead to a better overall anchor (i.e. prevents the algorithm of getting stuck in a local maximum) at the expense of more computation time. |
| `stop_on_first` | `bool` | `True` | If ``True``, the beam search algorithm will return the first anchor that has satisfies the probability constraint. |
| `max_anchor_size` | `Optional[int]` | `None` | Maximum number of features to include in an anchor. |
| `min_samples_start` | `int` | `100` | Number of samples used for anchor search initialisation. |
| `n_covered_ex` | `int` | `10` | How many examples where anchors apply to store for each anchor sampled during search (both examples where prediction on samples agrees/disagrees with predicted label are stored). |
| `binary_cache_size` | `int` | `10000` | The anchor search pre-allocates `binary_cache_size` batches for storing the boolean arrays returned during sampling. |
| `cache_margin` | `int` | `1000` | When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache of the same size is pre-allocated to continue buffering samples. |
| `verbose` | `bool` | `False` | Display updates during the anchor search iterations. |
| `verbose_every` | `int` | `1` | Frequency of displayed iterations during anchor search process. |
| `**kwargs` |  |  | Other keyword arguments passed to the anchor beam search and the text sampling and perturbation functions. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New predictor function. |

**Returns**
- Type: `None`

#### `sampler`

```python
sampler(anchor: Tuple[int, tuple], num_samples: int, compute_labels: bool = True) -> Union[List[Union[numpy.ndarray, float, int]], List[numpy.ndarray]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `Tuple[int, tuple]` |  | - ``int`` - the position of the anchor in the input batch. - ``tuple`` - the anchor itself, a list of words to be kept unchanged. |
| `num_samples` | `int` |  | Number of generated perturbed samples. |
| `compute_labels` | `bool` | `True` | If ``True``, an array of comparisons between predictions on perturbed samples and instance to be explained is returned. |

**Returns**
- Type: `Union[List[Union[numpy.ndarray, float, int]], List[numpy.ndarray]]`
