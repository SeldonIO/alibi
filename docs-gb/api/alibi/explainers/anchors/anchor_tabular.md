# `alibi.explainers.anchors.anchor_tabular`
## Classes
### `AnchorTabular` (_inherits from `Explainer`, `FitMixin`, `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

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

#### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `predictor` | `Optional[Callable]` |  |

#### Methods

##### `add_names_to_exp`

```python
add_names_to_exp(explanation: dict) -> None
```

Add feature names to explanation dictionary.

Parameters
----------
explanation
    Dict with anchors and additional metadata.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explanation` | `dict` |  |  |

**Returns**
- Type: `None`

##### `explain`

```python
explain(X: numpy.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

Explain prediction made by classifier on instance `X`.

Parameters
----------
X
    Instance to be explained.
threshold
    Minimum anchor precision threshold. The algorithm tries to find an anchor that maximizes the coverage
    under precision constraint. The precision constraint is formally defined as
    :math:`P(prec(A) \ge t) \ge 1 - \delta`, where :math:`A` is an anchor, :math:`t` is the `threshold`
    parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision
    of an anchor. In other words, we are seeking for an anchor having its precision greater or equal than
    the given `threshold` with a confidence of `(1 - delta)`. A higher value guarantees that the anchors are
    faithful to the model, but also leads to more computation time. Note that there are cases in which the
    precision constraint cannot be satisfied due to the quantile-based discretisation of the numerical
    features. If that is the case, the best (i.e. highest coverage) non-eligible anchor is returned.
delta
    Significance threshold. `1 - delta` represents the confidence threshold for the anchor precision
    (see `threshold`) and the selection of the best anchor candidate in each iteration (see `tau`).
tau
    Multi-armed bandit parameter used to select candidate anchors in each iteration. The multi-armed bandit
    algorithm tries to find within a tolerance `tau` the most promising (i.e. according to the precision)
    `beam_size` candidate anchor(s) from a list of proposed anchors. Formally, when the `beam_size=1`,
    the multi-armed bandit algorithm seeks to find an anchor :math:`A` such that
    :math:`P(prec(A) \ge prec(A^\star) - \tau) \ge 1 - \delta`, where :math:`A^\star` is the anchor
    with the highest true precision (which we don't know), :math:`\tau` is the `tau` parameter,
    :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor.
    In other words, in each iteration, the algorithm returns with a probability of at least `1 - delta` an
    anchor :math:`A` with a precision within an error tolerance of `tau` from the precision of the
    highest true precision anchor :math:`A^\star`. A bigger value for `tau` means faster convergence but also
    looser anchor conditions.
batch_size
    Batch size used for sampling. The Anchor algorithm will query the black-box model in batches of size
    `batch_size`. A larger `batch_size` gives more confidence in the anchor, again at the expense of
    computation time since it involves more model prediction calls.
coverage_samples
    Number of samples used to estimate coverage from during result search.
beam_size
    Number of candidate anchors selected by the multi-armed bandit algorithm in each iteration from a list of
    proposed anchors. A bigger beam  width can lead to a better overall anchor (i.e. prevents the algorithm
    of getting stuck in a local maximum) at the expense of more computation time.
stop_on_first
    If ``True``, the beam search algorithm will return the first anchor that has satisfies the
    probability constraint.
max_anchor_size
    Maximum number of features in result.
min_samples_start
    Min number of initial samples.
n_covered_ex
    How many examples where anchors apply to store for each anchor sampled during search
    (both examples where prediction on samples agrees/disagrees with `desired_label` are stored).
binary_cache_size
    The result search pre-allocates `binary_cache_size` batches for storing the binary arrays
    returned during sampling.
cache_margin
    When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache
    of the same size is pre-allocated to continue buffering samples.
verbose
    Display updates during the anchor search iterations.
verbose_every
    Frequency of displayed iterations during anchor search process.

Returns
-------
explanation
    `Explanation` object containing the result explaining the instance with additional metadata as attributes.
    See usage at `AnchorTabular examples`_ for details.

    .. _AnchorTabular examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html

Raises
------
:py:class:`alibi.exceptions.NotFittedError`
    If `fit` has not been called prior to calling `explain`.

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
| `kwargs` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `fit`

```python
fit(train_data: numpy.ndarray, disc_perc: Tuple[Union[int, float], .Ellipsis] = (25, 50, 75), kwargs) -> alibi.explainers.anchors.anchor_tabular.AnchorTabular
```

Fit discretizer to train data to bin numerical features into ordered bins and compute statistics for

numerical features. Create a mapping between the bin numbers of each discretised numerical feature and the
row id in the training set where it occurs.

Parameters
----------
train_data
    Representative sample from the training data.
disc_perc
    List with percentiles (`int`) used for discretization.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  |  |
| `disc_perc` | `Tuple[Union[int, float], .Ellipsis]` | `(25, 50, 75)` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `alibi.explainers.anchors.anchor_tabular.AnchorTabular`

##### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

Parameters
----------
predictor
    New predictor function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

### `TabularSampler`

A sampler that uses an underlying training set to draw records that have a subset of features with

values specified in an instance to be explained, `X`.

#### Constructor

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

#### Methods

##### `build_lookups`

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
---------
X
    Instance to be explained.

Returns
-------
A list containing three dictionaries, whose keys are encoded feature IDs

 - `cat_lookup` - maps categorical variables to their value in `X`.

 - `ord_lookup` - maps discretized numerical variables to the bins they can be sampled from given `X`.

 - `enc2feat_idx` - maps the encoded IDs to the original (training set) feature column IDs.

Notes
-----
Each continuous variable has `n_bins - 1` corresponding entries in `ord_lookup`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `List[Dict]`

##### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

Compute the agreement between a classifier prediction on an instance to be explained and the

prediction on a set of samples which have a subset of features fixed to specific values.

Parameters
----------
samples
    Samples whose labels are to be compared with the instance label.

Returns
-------
An array of integers indicating whether the prediction was the same as the instance label.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

##### `deferred_init`

```python
deferred_init(train_data: Union[numpy.ndarray, typing.Any], d_train_data: Union[numpy.ndarray, typing.Any]) -> typing.Any
```

Initialise the tabular sampler object with data, discretizer, feature statistics and

build an index from feature values and bins to database rows for each feature.

Parameters
----------
train_data:
    Data from which samples are drawn. Can be a `numpy` array or a `ray` future.
d_train_data:
    Discretized version for training data. Can be a `numpy` array or a `ray` future.

Returns
-------
An initialised sampler.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `Union[numpy.ndarray, typing.Any]` |  |  |
| `d_train_data` | `Union[numpy.ndarray, typing.Any]` |  |  |

**Returns**
- Type: `typing.Any`

##### `get_features_index`

```python
get_features_index(anchor: tuple) -> Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]
```

Given an anchor, this function finds the row indices in the training set where the feature has

the same value as the feature in the instance to be explained (for ordinal variables, the row
indices are those of rows which contain records with feature values in the same bin). The algorithm
uses both the feature *encoded* ids in anchor and the feature ids in the input data set. The two
are mapped by `self.enc2feat_idx`.

Parameters
----------
anchor
    The anchor for which the training set row indices are to be retrieved. The ints represent
    encoded feature ids.

Returns
-------
allowed_bins
    Maps original feature ids to the bins that the feature should be sampled from given the input anchor.
allowed_rows
    Maps original feature ids to the training set rows where these features have the same value as the anchor.
unk_feat_values
    When a categorical variable with the specified value/discretized variable in the specified bin is not found
    in the training set, a tuple is added to `unk_feat_values` to indicate the original feature id, its type
    (``'c'`` = categorical, ``'o'`` = discretized continuous) and the value/bin it should be sampled from.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |

**Returns**
- Type: `Tuple[Dict[int, set[int]], Dict[int, typing.Any], List[Tuple[int, str, Union[typing.Any, int]]]]`

##### `handle_unk_features`

```python
handle_unk_features(allowed_bins: Dict[int, set[int]], num_samples: int, samples: numpy.ndarray, unk_feature_values: List[Tuple[int, str, Union[typing.Any, int]]]) -> None
```

Replaces unknown feature values with defaults. For categorical variables, the replacement value is

the same as the value of the unknown feature. For continuous variables, a value is sampled uniformly
at random from the feature range.

Parameters
----------
allowed_bins
    See :py:meth:`alibi.explainers.anchors.anchor_tabular.TabularSampler.get_features_index` method.
num_samples
    Number of replacement values.
samples
    Contains the samples whose values are to be replaced.
unk_feature_values
    List of tuples where: [0] is original feature id, [1] feature type, [2] if var is categorical,
    replacement value, otherwise None

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `allowed_bins` | `Dict[int, set[int]]` |  |  |
| `num_samples` | `int` |  |  |
| `samples` | `numpy.ndarray` |  |  |
| `unk_feature_values` | `List[Tuple[int, str, Union[typing.Any, int]]]` |  |  |

**Returns**
- Type: `None`

##### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray, float]
```

Implements functionality described in

:py:meth:`alibi.explainers.anchors.anchor_tabular.TabularSampler.__call__`.

Parameters
----------
anchor:
    Each int is an encoded feature id.
num_samples
    Number of samples.

Returns
-------
samples
    Sampled data from training set.
d_samples
    Like samples, but continuous data is converted to ordinal discrete data (binned).
coverage
    The coverage of the result in the training data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |
| `num_samples` | `int` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray, float]`

##### `replace_features`

```python
replace_features(samples: numpy.ndarray, allowed_rows: Dict[int, typing.Any], uniq_feat_ids: List[int], partial_anchor_rows: List[numpy.ndarray], nb_partial_anchors: numpy.ndarray, num_samples: int) -> None
```

The method creates perturbed samples by first replacing all partial anchors with partial anchors drawn

from the training set. Then remainder of the features are then replaced with random values drawn from
the same bin for discretized continuous features and same value for categorical features.

Parameters
----------
samples
    Randomly drawn samples, where the anchor does not apply.
allowed_rows
    Maps feature ids to the rows indices in training set where the feature has same value as instance (cat.)
    or is in the same bin.
uniq_feat_ids
    Multiple encoded features in the anchor can map to the same original feature id. Unique features in the
    anchor. This is the list of unique original features id in the anchor.
partial_anchor_rows
    The rows in the training set where each partial anchor applies. Last entry is an array of row indices where
    the entire anchor applies.
nb_partial_anchors
    The number of training records which contain each partial anchor.
num_samples
    Number of perturbed samples to be returned.

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

##### `set_instance_label`

```python
set_instance_label(X: numpy.ndarray) -> None
```

Sets the sampler label. Necessary for setting the remote sampling process state during explain call.

Parameters
----------
X
    Instance to be explained.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `None`

##### `set_n_covered`

```python
set_n_covered(n_covered: int) -> None
```

Set the number of examples to be saved for each result and partial result during search process.

The same number of examples is saved in the case where the predictions on perturbed samples and
original instance agree or disagree.

Parameters
---------
n_covered
    Number of examples to be saved.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_covered` | `int` |  |  |

**Returns**
- Type: `None`
