# `alibi.explainers.anchors.anchor_tabular`
## `AnchorTabular`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
AnchorTabular(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], feature_names: List[str], categorical_names: Optional[Dict[int, List[str]]] = None, dtype: Type[numpy.generic] = <class 'numpy.float32'>, ohe: bool = False, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `List[str]` |  |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` |  |
| `ohe` | `bool` | `False` |  |
| `seed` | `Optional[int]` | `None` |  |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `predictor` | `Optional[Callable]` |  |

### Methods

#### `add_names_to_exp`

```python
add_names_to_exp(explanation: dict) -> None
```

Add feature names to explanation dictionary.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explanation` | `dict` |  |  |

**Returns**
- Type: `None`

#### `explain`

```python
explain(X: numpy.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

Explain prediction made by classifier on instance `X`.

Parameters

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
| `min_samples_start` | `int` | `100` |  |
| `n_covered_ex` | `int` | `10` |  |
| `binary_cache_size` | `int` | `10000` |  |
| `cache_margin` | `int` | `1000` |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, disc_perc: Tuple[Union[int, float], .Ellipsis] = (25, 50, 75), kwargs) -> alibi.explainers.anchors.anchor_tabular.AnchorTabular
```

Fit discretizer to train data to bin numerical features into ordered bins and compute statistics for

numerical features. Create a mapping between the bin numbers of each discretised numerical feature and the
row id in the training set where it occurs.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  |  |
| `disc_perc` | `Tuple[Union[int, float], .Ellipsis]` | `(25, 50, 75)` |  |

**Returns**
- Type: `alibi.explainers.anchors.anchor_tabular.AnchorTabular`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

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
| `predictor` | `Callable` |  |  |
| `disc_perc` | `Tuple[Union[int, float], .Ellipsis]` |  |  |
| `numerical_features` | `List[int]` |  |  |
| `categorical_features` | `List[int]` |  |  |
| `feature_names` | `list` |  |  |
| `feature_values` | `dict` |  |  |
| `n_covered_ex` | `int` | `10` |  |
| `seed` | `Optional[int]` | `None` |  |

### Methods

#### `build_lookups`

```python
build_lookups(X: numpy.ndarray) -> List[Dict]
```

An encoding of the feature IDs is created by assigning each bin of a discretized numerical variable and each

categorical variable a unique index. For a dataset containing, e.g., a numerical variable with 5 bins and
3 categorical variables, indices 0 - 4 represent bins of the numerical variable whereas indices 5, 6, 7
represent the encoded indices of the categorical variables (but see note for caviats). The encoding is
necessary so that the different ranges of the numerical variable can be sampled during result construction.
Note that the encoded indices represent the predicates used during the anchor construction process (i.e., and
anchor is a collection of encoded indices.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `List[Dict]`

#### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

Compute the agreement between a classifier prediction on an instance to be explained and the

prediction on a set of samples which have a subset of features fixed to specific values.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

#### `deferred_init`

```python
deferred_init(train_data: Union[numpy.ndarray, typing.Any], d_train_data: Union[numpy.ndarray, typing.Any]) -> typing.Any
```

Initialise the tabular sampler object with data, discretizer, feature statistics and

build an index from feature values and bins to database rows for each feature.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `Union[numpy.ndarray, typing.Any]` |  |  |
| `d_train_data` | `Union[numpy.ndarray, typing.Any]` |  |  |

**Returns**
- Type: `typing.Any`

#### `get_features_index`

```python
get_features_index(anchor: tuple) -> Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]
```

Given an anchor, this function finds the row indices in the training set where the feature has

the same value as the feature in the instance to be explained (for ordinal variables, the row
indices are those of rows which contain records with feature values in the same bin). The algorithm
uses both the feature *encoded* ids in anchor and the feature ids in the input data set. The two
are mapped by `self.enc2feat_idx`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |

**Returns**
- Type: `Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]`

#### `handle_unk_features`

```python
handle_unk_features(allowed_bins: Dict[int, set[int]], num_samples: int, samples: numpy.ndarray, unk_feature_values: List[Tuple[int, str, Union[typing.Any, int]]]) -> None
```

Replaces unknown feature values with defaults. For categorical variables, the replacement value is

the same as the value of the unknown feature. For continuous variables, a value is sampled uniformly
at random from the feature range.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `allowed_bins` | `Dict[int, set[int]]` |  |  |
| `num_samples` | `int` |  |  |
| `samples` | `numpy.ndarray` |  |  |
| `unk_feature_values` | `List[Tuple[int, str, Union[typing.Any, int]]]` |  |  |

**Returns**
- Type: `None`

#### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray, float]
```

Implements functionality described in

:py:meth:`alibi.explainers.anchors.anchor_tabular.TabularSampler.__call__`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |
| `num_samples` | `int` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray, float]`

#### `replace_features`

```python
replace_features(samples: numpy.ndarray, allowed_rows: Dict[int, typing.Any], uniq_feat_ids: List[int], partial_anchor_rows: List[numpy.ndarray], nb_partial_anchors: numpy.ndarray, num_samples: int) -> None
```

The method creates perturbed samples by first replacing all partial anchors with partial anchors drawn

from the training set. Then remainder of the features are then replaced with random values drawn from
the same bin for discretized continuous features and same value for categorical features.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |
| `allowed_rows` | `Dict[int, typing.Any]` |  |  |
| `uniq_feat_ids` | `List[int]` |  |  |
| `partial_anchor_rows` | `List[numpy.ndarray]` |  |  |
| `nb_partial_anchors` | `numpy.ndarray` |  |  |
| `num_samples` | `int` |  |  |

**Returns**
- Type: `None`

#### `set_instance_label`

```python
set_instance_label(X: numpy.ndarray) -> None
```

Sets the sampler label. Necessary for setting the remote sampling process state during explain call.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `None`

#### `set_n_covered`

```python
set_n_covered(n_covered: int) -> None
```

Set the number of examples to be saved for each result and partial result during search process.

The same number of examples is saved in the case where the predictions on perturbed samples and
original instance agree or disagree.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_covered` | `int` |  |  |

**Returns**
- Type: `None`
