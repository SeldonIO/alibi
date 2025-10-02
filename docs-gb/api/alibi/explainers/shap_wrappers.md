# `alibi.explainers.shap_wrappers`
## Classes
### `KernelExplainerWrapper` (_inherits from `KernelExplainer`, `Explainer`, `Serializable`)

A wrapper around `shap.KernelExplainer` that supports:

- fixing the seed when instantiating the KernelExplainer in a separate process.

    - passing a batch index to the explainer so that a parallel explainer pool can return batches in         arbitrary order.

#### Constructor

```python
KernelExplainerWrapper(self, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

#### Methods

##### `get_explanation`

```python
get_explanation(X: Union[Tuple[int, numpy.ndarray], numpy.ndarray], kwargs) -> Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]
```

Wrapper around `shap.KernelExplainer.shap_values` that allows calling the method with a tuple containing a

batch index and a batch of instances.

Parameters
----------
X
    When called from a distributed context, it is a tuple containing a batch index and a batch to be explained.
    Otherwise, it is an array of instances to be explained.
**kwargs
    `shap.KernelExplainer.shap_values` kwarg values.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[Tuple[int, numpy.ndarray], numpy.ndarray]` |  |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]`

##### `return_attribute`

```python
return_attribute(name: str) -> typing.Any
```

Returns an attribute specified by its name. Used in a distributed context where the actor properties cannot be

accessed using the dot syntax.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `name` | `str` |  |  |

**Returns**
- Type: `typing.Any`

### `KernelShap` (_inherits from `Explainer`, `FitMixin`, `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

```python
KernelShap(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], link: str = 'identity', feature_names: Union[List[str], Tuple[str], NoneType] = None, categorical_names: Optional[Dict[int, List[str]]] = None, task: str = 'classification', seed: Optional[int] = None, distributed_opts: Optional[Dict] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `link` | `str` | `'identity'` |  |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `task` | `str` | `'classification'` |  |
| `seed` | `Optional[int]` | `None` |  |
| `distributed_opts` | `Optional[Dict]` | `None` |  |

#### Methods

##### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> alibi.api.interfaces.Explanation
```

Explains the instances in the array `X`.

Parameters
----------
X
    Instances to be explained.
summarise_result
    Specifies whether the shap values corresponding to dimensions of encoded categorical variables should be
    summed so that a single shap value is returned for each categorical variable. Both the start indices of
    the categorical variables (`cat_vars_start_idx`) and the encoding dimensions (`cat_vars_enc_dim`)
    have to be specified
cat_vars_start_idx
    The start indices of the categorical variables. If specified, `cat_vars_enc_dim` should also be specified.
cat_vars_enc_dim
    The length of the encoding dimension for each categorical variable. If specified `cat_vars_start_idx` should
    also be specified.
**kwargs
    Keyword arguments specifying explain behaviour. Valid arguments are:

        - `nsamples` - controls the number of predictor calls and therefore runtime.

        - `l1_reg` - the algorithm is exponential in the feature dimension. If set to `auto` the algorithm will                 first run a feature selection algorithm to select the top features, provided the fraction of sampled                 sets of missing features is less than 0.2 from the number of total subsets. The Akaike Information                 Criterion is used in this case. See our examples for more details about available settings for this                 parameter. Note that by first running a feature selection step, the shapley values of the remainder of                 the features will be different to those estimated from the entire set.

    For more details, please see the shap library `documentation`_ .

        .. _documentation:
           https://shap.readthedocs.io/en/stable/.

Returns
-------
explanation
    An explanation object containing the shap values and prediction in the `data` field, along with a `meta`
    field containing additional data. See usage at `KernelSHAP examples`_ for details.

    .. _KernelSHAP examples:
       https://docs.seldon.io/projects/alibi/en/stable/methods/KernelSHAP.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix]` |  |  |
| `summarise_result` | `bool` | `False` |  |
| `cat_vars_start_idx` | `Optional[Sequence[int]]` | `None` |  |
| `cat_vars_enc_dim` | `Optional[Sequence[int]]` | `None` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `fit`

```python
fit(background_data: Union[numpy.ndarray, scipy.sparse._matrix.spmatrix, pandas.core.frame.DataFrame, shap.utils._legacy.Data], summarise_background: Union[bool, str] = False, n_background_samples: int = 300, group_names: Union[List[str], Tuple[str], None] = None, groups: Optional[List[Union[Tuple[int], List[int]]]] = None, weights: Union[List[float], Tuple[float], numpy.ndarray, None] = None, kwargs) -> alibi.explainers.shap_wrappers.KernelShap
```

This takes a background dataset (usually a subsample of the training set) as an input along with several

user specified options and initialises a `KernelShap` explainer. The runtime of the algorithm depends on the
number of samples in this dataset and on the number of features in the dataset. To reduce the size of the
dataset, the `summarise_background` option and `n_background_samples` should be used. To reduce the feature
dimensionality, encoded categorical variables can be treated as one during the feature perturbation process;
this decreases the effective feature dimensionality, can reduce the variance of the shap values estimation and
reduces slightly the number of calls to the predictor. Further runtime savings can be achieved by changing the
`nsamples` parameter in the call to explain. Runtime reduction comes with an accuracy trade-off, so it is better
to experiment with a runtime reduction method and understand results stability before using the system.

Parameters
-----------
background_data
    Data used to estimate feature contributions and baseline values for force plots. The rows of the
    background data should represent samples and the columns features.
summarise_background
    A large background dataset impacts the runtime and memory footprint of the algorithm. By setting
    this argument to ``True``, only `n_background_samples` from the provided data are selected. If
    group_names or groups arguments are specified, the algorithm assumes that the data contains categorical
    variables so the records are selected uniformly at random. Otherwise, `shap.kmeans` (a wrapper
    around `sklearn` k-means implementation) is used for selection. If set to ``'auto'``, a default of
    `KERNEL_SHAP_BACKGROUND_THRESHOLD` samples is selected.
n_background_samples
    The number of samples to keep in the background dataset if ``summarise_background=True``.
groups:
    A list containing sub-lists specifying the indices of features belonging to the same group.
group_names:
    If specified, this array is used to treat groups of features as one during feature perturbation.
    This feature can be useful, for example, to treat encoded categorical variables as one and can
    result in computational savings (this may require adjusting the `nsamples` parameter).
weights:
    A sequence or array of weights. This is used only if grouping is specified and assigns a weight
    to each point in the dataset.
**kwargs:
    Expected keyword arguments include `keep_index` (bool) and should be used if a data frame containing an
    index column is passed to the algorithm.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, scipy.sparse._matrix.spmatrix, pandas.core.frame.DataFrame, shap.utils._legacy.Data]` |  |  |
| `summarise_background` | `Union[bool, str]` | `False` |  |
| `n_background_samples` | `int` | `300` |  |
| `group_names` | `Union[List[str], Tuple[str], None]` | `None` |  |
| `groups` | `Optional[List[Union[Tuple[int], List[int]]]]` | `None` |  |
| `weights` | `Union[List[float], Tuple[float], numpy.ndarray, None]` | `None` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `alibi.explainers.shap_wrappers.KernelShap`

##### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the prediction function.

Parameters
----------
predictor
    New prediction function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

### `TreeShap` (_inherits from `Explainer`, `FitMixin`, `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

```python
TreeShap(self, predictor: Any, model_output: str = 'raw', feature_names: Union[List[str], Tuple[str], NoneType] = None, categorical_names: Optional[Dict[int, List[str]]] = None, task: str = 'classification', seed: Optional[int] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  |  |
| `model_output` | `str` | `'raw'` |  |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` |  |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` |  |
| `task` | `str` | `'classification'` |  |
| `seed` | `Optional[int]` | `None` |  |

#### Methods

##### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, ForwardRef('catboost.Pool')], y: Optional[numpy.ndarray] = None, interactions: bool = False, approximate: bool = False, check_additivity: bool = True, tree_limit: Optional[int] = None, summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> Explanation
```

Explains the instances in `X`. `y` should be passed if the model loss function is to be explained,

which can be useful in order to understand how various features affect model performance over
time. This is only possible if the explainer has been fitted with a background dataset and
requires setting `model_output='log_loss'`.

Parameters
----------
X
    Instances to be explained.
y
    Labels corresponding to rows of `X`. Should be passed only if a background dataset was passed to the
    `fit` method.
interactions
    If ``True``, the shap value for every feature of every instance in `X` is decomposed into
    `X.shape[1] - 1` shap value interactions and one main effect. This is only supported if `fit` is called
    with `background_dataset=None`.
approximate
    If ``True``, an approximation to the shap values that does not account for feature order is computed. This
    was proposed by `Ando Sabaas`_ here . Check `this`_ resource for more details. This option is currently
    only supported for `xgboost` and `sklearn` models.

    .. _Ando Sabaas:
       https://github.com/andosa/treeinterpreter

    .. _this:
       https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-019-0138-9/MediaObjects/42256_2019_138_MOESM1_ESM.pdf

check_additivity
    If ``True``, output correctness is ensured if ``model_output='raw'`` has been passed to the constructor.
tree_limit
    Explain the output of a subset of the first `tree_limit` trees in an ensemble model.
summarise_result
    This should be set to ``True`` only when some of the columns in `X` represent encoded dimensions of a
    categorical variable and one single shap value per categorical variable is desired. Both
    `cat_vars_start_idx` and `cat_vars_enc_dim` should be specified as detailed below to allow this.
cat_vars_start_idx
    The start indices of the categorical variables.
cat_vars_enc_dim
    The length of the encoding dimension for each categorical variable.

Returns
-------
explanation
    An `Explanation` object containing the shap values and prediction in the `data` field, along with a
    `meta` field containing additional data. See usage at `TreeSHAP examples`_ for details.

    .. _TreeSHAP examples:
       https://docs.seldon.io/projects/alibi/en/stable/methods/TreeSHAP.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, ForwardRef('catboost.Pool')]` |  |  |
| `y` | `Optional[numpy.ndarray]` | `None` |  |
| `interactions` | `bool` | `False` |  |
| `approximate` | `bool` | `False` |  |
| `check_additivity` | `bool` | `True` |  |
| `tree_limit` | `Optional[int]` | `None` |  |
| `summarise_result` | `bool` | `False` |  |
| `cat_vars_start_idx` | `Optional[Sequence[int]]` | `None` |  |
| `cat_vars_enc_dim` | `Optional[Sequence[int]]` | `None` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `Explanation`

##### `fit`

```python
fit(background_data: Union[numpy.ndarray, pandas.core.frame.DataFrame, None] = None, summarise_background: Union[bool, str] = False, n_background_samples: int = 1000, kwargs) -> alibi.explainers.shap_wrappers.TreeShap
```

This function instantiates an explainer which can then be use to explain instances using the `explain` method.

If no background dataset is passed, the explainer uses the path-dependent feature perturbation algorithm
to explain the values. As such, only the model raw output can be explained and this should be reflected by
passing ``model_output='raw'`` when instantiating the explainer. If a background dataset is passed, the
interventional feature perturbation algorithm is used. Using this algorithm, probability outputs can also be
explained. Additionally, if the ``model_output='log_loss'`` option is passed to the explainer constructor, then
the model loss function can be explained by passing the labels as the `y` argument to the explain method.
A limited number of loss functions are supported, as detailed in the constructor documentation.

Parameters
-----------
background_data
    Data used to estimate feature contributions and baseline values for force plots. The rows of the
    background data should represent samples and the columns features.
summarise_background
    A large background dataset may impact the runtime and memory footprint of the algorithm. By setting
    this argument to ``True``, only `n_background_samples` from the provided data are selected. If the
    `categorical_names` argument has been passed to the constructor, subsampling of the data is used.
    Otherwise, `shap.kmeans` (a wrapper around `sklearn.kmeans` implementation) is used for selection.
    If set to ``'auto'``, a default of `TREE_SHAP_BACKGROUND_WARNING_THRESHOLD` samples is selected.
n_background_samples
    The number of samples to keep in the background dataset if ``summarise_background=True``.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, None]` | `None` |  |
| `summarise_background` | `Union[bool, str]` | `False` |  |
| `n_background_samples` | `int` | `1000` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `alibi.explainers.shap_wrappers.TreeShap`

##### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

Resets the predictor.

Parameters
----------
predictor
    New prediction.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  |  |

**Returns**
- Type: `None`

## Functions
### `rank_by_importance`

```python
rank_by_importance(shap_values: List[numpy.ndarray], feature_names: Union[List[str], Tuple[str], None] = None) -> Dict
```

Given the shap values estimated for a multi-output model, this function ranks

features according to their importance. The feature importance is the average
absolute value for a given feature.

Parameters
----------
shap_values
    Each element corresponds to a `samples x features` array of shap values corresponding
    to each model output.
feature_names
    Each element is the name of the column with the corresponding index in each of the
    arrays in the `shap_values` list.

Returns
-------
importances
    A dictionary of the form::

        {
            '0': {'ranked_effect': array([0.2, 0.5, ...]), 'names': ['feat_3', 'feat_5', ...]},
            '1': {'ranked_effect': array([0.3, 0.2, ...]), 'names': ['feat_6', 'feat_1', ...]},
            ...
            'aggregated': {'ranked_effect': array([0.9, 0.7, ...]), 'names': ['feat_3', 'feat_6', ...]}
        }

    The keys of the first level represent the index of the model output. The feature effects in
    `ranked_effect` and the corresponding feature names in `names` are sorted from highest (most
    important) to lowest (least important). The values in the `aggregated` field are obtained by
    summing the shap values for all the model outputs and then computing the effects. Given an
    output, the effects are defined as the average magnitude of the shap values across the instances
    to be explained.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shap_values` | `List[numpy.ndarray]` |  |  |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` |  |

**Returns**
- Type: `Dict`

### `sum_categories`

```python
sum_categories(values: numpy.ndarray, start_idx: Sequence[int], enc_feat_dim: Sequence[int])
```

This function is used to reduce specified slices in a two- or three- dimensional array.

For two-dimensional `values` arrays, for each entry in `start_idx`, the function sums the
following `k` columns where `k` is the corresponding entry in the `enc_feat_dim` sequence.
The columns whose indices are not in `start_idx` are left unchanged. This arises when the slices
contain the shap values for each dimension of an encoded categorical variable and a single shap
value for each variable is desired.

For three-dimensional `values` arrays, the reduction is applied for each rank 2 subarray, first along
the column dimension and then across the row dimension. This arises when summarising shap interaction values.
Each rank 2 array is a `E x E` matrix of shap interaction values, where `E` is the dimension of the data after
one-hot encoding. The result of applying the reduction yields a rank 2 array of dimension `F x F`, where `F` is the
number of features (i.e., the feature dimension of the data matrix before encoding). By applying this
transformation, a single value describing the interaction of categorical features i and j and a single value
describing the interaction of `j` and `i` is returned.

Parameters
----------
values
    A two or three dimensional array to be reduced, as described above.
start_idx
    The start indices of the columns to be summed.
enc_feat_dim
    The number of columns to be summed, one for each start index.

Returns
-------
new_values
    An array whose columns have been summed according to the entries in `start_idx` and `enc_feat_dim`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |
| `start_idx` | `Sequence[int]` |  |  |
| `enc_feat_dim` | `Sequence[int]` |  |  |
