# `alibi.explainers.anchors.anchor_tabular`
## Constants
### `DEFAULT_DATA_ANCHOR`
```python
DEFAULT_DATA_ANCHOR: dict = {'anchor': [], 'precision': None, 'coverage': None, 'raw': None}
```

### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

## `AnchorTabular`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
AnchorTabular(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], feature_names: List[str], categorical_names: Optional[Dict[int, List[str]]] = None, dtype: Type[numpy.generic] = <class 'numpy.float32'>, ohe: bool = False, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. |
| `feature_names` | `List[str]` |  | List with feature names. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Dictionary where keys are feature columns and values are the categories for the feature. |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` | A `numpy` scalar type that corresponds to the type of input array expected by `predictor`. This may be used to construct arrays of the given type to be passed through the `predictor`. For most use cases this argument should have no effect, but it is exposed for use with predictors that would break when called with an array of unsupported type. |
| `ohe` | `bool` | `False` | Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are assumed to have ordinal encodings. |
| `seed` | `Optional[int]` | `None` | Used to set the random number generator for repeatability purposes. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `predictor` | `Optional[Callable]` |  |

### Methods

#### `add_names_to_exp`

```python
add_names_to_exp(explanation: dict) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explanation` | `dict` |  | Dict with anchors and additional metadata. |

**Returns**
- Type: `None`

#### `explain`

```python
explain(X: numpy.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to be explained. |
| `threshold` | `float` | `0.95` | Minimum anchor precision threshold. The algorithm tries to find an anchor that maximizes the coverage under precision constraint. The precision constraint is formally defined as :math:`P(prec(A) \ge t) \ge 1 - \delta`, where :math:`A` is an anchor, :math:`t` is the `threshold` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, we are seeking for an anchor having its precision greater or equal than the given `threshold` with a confidence of `(1 - delta)`. A higher value guarantees that the anchors are faithful to the model, but also leads to more computation time. Note that there are cases in which the precision constraint cannot be satisfied due to the quantile-based discretisation of the numerical features. If that is the case, the best (i.e. highest coverage) non-eligible anchor is returned. |
| `delta` | `float` | `0.1` | Significance threshold. `1 - delta` represents the confidence threshold for the anchor precision (see `threshold`) and the selection of the best anchor candidate in each iteration (see `tau`). |
| `tau` | `float` | `0.15` | Multi-armed bandit parameter used to select candidate anchors in each iteration. The multi-armed bandit algorithm tries to find within a tolerance `tau` the most promising (i.e. according to the precision) `beam_size` candidate anchor(s) from a list of proposed anchors. Formally, when the `beam_size=1`, the multi-armed bandit algorithm seeks to find an anchor :math:`A` such that :math:`P(prec(A) \ge prec(A^\star) - \tau) \ge 1 - \delta`, where :math:`A^\star` is the anchor with the highest true precision (which we don't know), :math:`\tau` is the `tau` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, in each iteration, the algorithm returns with a probability of at least `1 - delta` an anchor :math:`A` with a precision within an error tolerance of `tau` from the precision of the highest true precision anchor :math:`A^\star`. A bigger value for `tau` means faster convergence but also looser anchor conditions. |
| `batch_size` | `int` | `100` | Batch size used for sampling. The Anchor algorithm will query the black-box model in batches of size `batch_size`. A larger `batch_size` gives more confidence in the anchor, again at the expense of computation time since it involves more model prediction calls. |
| `coverage_samples` | `int` | `10000` | Number of samples used to estimate coverage from during result search. |
| `beam_size` | `int` | `1` | Number of candidate anchors selected by the multi-armed bandit algorithm in each iteration from a list of proposed anchors. A bigger beam  width can lead to a better overall anchor (i.e. prevents the algorithm of getting stuck in a local maximum) at the expense of more computation time. |
| `stop_on_first` | `bool` | `False` | If ``True``, the beam search algorithm will return the first anchor that has satisfies the probability constraint. |
| `max_anchor_size` | `Optional[int]` | `None` | Maximum number of features in result. |
| `min_samples_start` | `int` | `100` | Min number of initial samples. |
| `n_covered_ex` | `int` | `10` | How many examples where anchors apply to store for each anchor sampled during search (both examples where prediction on samples agrees/disagrees with `desired_label` are stored). |
| `binary_cache_size` | `int` | `10000` | The result search pre-allocates `binary_cache_size` batches for storing the binary arrays returned during sampling. |
| `cache_margin` | `int` | `1000` | When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache of the same size is pre-allocated to continue buffering samples. |
| `verbose` | `bool` | `False` | Display updates during the anchor search iterations. |
| `verbose_every` | `int` | `1` | Frequency of displayed iterations during anchor search process. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, disc_perc: Tuple[Union[int, float], .Ellipsis] = (25, 50, 75), kwargs) -> alibi.explainers.anchors.anchor_tabular.AnchorTabular
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  | Representative sample from the training data. |
| `disc_perc` | `Tuple[Union[int, float], .Ellipsis]` | `(25, 50, 75)` | List with percentiles (`int`) used for discretization. |

**Returns**
- Type: `alibi.explainers.anchors.anchor_tabular.AnchorTabular`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New predictor function. |

**Returns**
- Type: `None`

## `TabularSampler`

A sampler that uses an underlying training set to draw records that have a subset of features with
values specified in an instance to be explained, `X`.

### Constructor

```python
TabularSampler(self, predictor: Callable, disc_perc: Tuple[Union[int, float], ...], numerical_features: List[int], categorical_features: List[int], feature_names: list, feature_values: dict, n_covered_ex: int = 10, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | A callable that takes a tensor of `N` data points as inputs and returns `N` outputs. |
| `disc_perc` | `Tuple[Union[int, float], .Ellipsis]` |  | Percentiles used for numerical feature discretisation. |
| `numerical_features` | `List[int]` |  | Numerical features column IDs. |
| `categorical_features` | `List[int]` |  | Categorical features column IDs. |
| `feature_names` | `list` |  | Feature names. |
| `feature_values` | `dict` |  | Key: categorical feature column ID, value: values for the feature. |
| `n_covered_ex` | `int` | `10` | For each result, a number of samples where the prediction agrees/disagrees with the prediction on instance to be explained are stored. |
| `seed` | `Optional[int]` | `None` | If set, fixes the random number sequence. |

### Methods

#### `build_lookups`

```python
build_lookups(X: numpy.ndarray) -> List[Dict]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to be explained. |

**Returns**
- Type: `List[Dict]`

#### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  | Samples whose labels are to be compared with the instance label. |

**Returns**
- Type: `numpy.ndarray`

#### `deferred_init`

```python
deferred_init(train_data: Union[numpy.ndarray, typing.Any], d_train_data: Union[numpy.ndarray, typing.Any]) -> typing.Any
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `Union[numpy.ndarray, typing.Any]` |  | Data from which samples are drawn. Can be a `numpy` array or a `ray` future. |
| `d_train_data` | `Union[numpy.ndarray, typing.Any]` |  | Discretized version for training data. Can be a `numpy` array or a `ray` future. |

**Returns**
- Type: `typing.Any`

#### `get_features_index`

```python
get_features_index(anchor: tuple) -> Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  | The anchor for which the training set row indices are to be retrieved. The ints represent encoded feature ids. |

**Returns**
- Type: `Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]`

#### `handle_unk_features`

```python
handle_unk_features(allowed_bins: Dict[int, set[int]], num_samples: int, samples: numpy.ndarray, unk_feature_values: List[Tuple[int, str, Union[typing.Any, int]]]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `allowed_bins` | `Dict[int, set[int]]` |  | See :py:meth:`alibi.explainers.anchors.anchor_tabular.TabularSampler.get_features_index` method. |
| `num_samples` | `int` |  | Number of replacement values. |
| `samples` | `numpy.ndarray` |  | Contains the samples whose values are to be replaced. |
| `unk_feature_values` | `List[Tuple[int, str, Union[typing.Any, int]]]` |  | List of tuples where: [0] is original feature id, [1] feature type, [2] if var is categorical, replacement value, otherwise None |

**Returns**
- Type: `None`

#### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  | Each int is an encoded feature id. |
| `num_samples` | `int` |  | Number of samples. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray, float]`

#### `replace_features`

```python
replace_features(samples: numpy.ndarray, allowed_rows: Dict[int, typing.Any], uniq_feat_ids: List[int], partial_anchor_rows: List[numpy.ndarray], nb_partial_anchors: numpy.ndarray, num_samples: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  | Randomly drawn samples, where the anchor does not apply. |
| `allowed_rows` | `Dict[int, typing.Any]` |  | Maps feature ids to the rows indices in training set where the feature has same value as instance (cat.) or is in the same bin. |
| `uniq_feat_ids` | `List[int]` |  | Multiple encoded features in the anchor can map to the same original feature id. Unique features in the anchor. This is the list of unique original features id in the anchor. |
| `partial_anchor_rows` | `List[numpy.ndarray]` |  | The rows in the training set where each partial anchor applies. Last entry is an array of row indices where the entire anchor applies. |
| `nb_partial_anchors` | `numpy.ndarray` |  | The number of training records which contain each partial anchor. |
| `num_samples` | `int` |  | Number of perturbed samples to be returned. |

**Returns**
- Type: `None`

#### `set_instance_label`

```python
set_instance_label(X: numpy.ndarray) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to be explained. |

**Returns**
- Type: `None`

#### `set_n_covered`

```python
set_n_covered(n_covered: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_covered` | `int` |  | Number of examples to be saved. |

**Returns**
- Type: `None`
