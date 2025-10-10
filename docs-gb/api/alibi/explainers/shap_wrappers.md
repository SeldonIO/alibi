# `alibi.explainers.shap_wrappers`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `DEFAULT_DATA_KERNEL_SHAP`
```python
DEFAULT_DATA_KERNEL_SHAP: dict = {'shap_values': [], 'expected_value': [], 'categorical_names': {}, 'feature_n...
```

### `DEFAULT_DATA_TREE_SHAP`
```python
DEFAULT_DATA_TREE_SHAP: dict = {'shap_values': [], 'shap_interaction_values': [], 'expected_value': [], 'cat...
```

### `DEFAULT_META_KERNEL_SHAP`
```python
DEFAULT_META_KERNEL_SHAP: dict = {'name': None, 'type': ['blackbox'], 'task': None, 'explanations': ['local', ...
```

### `DEFAULT_META_TREE_SHAP`
```python
DEFAULT_META_TREE_SHAP: dict = {'name': None, 'type': ['whitebox'], 'task': None, 'explanations': ['local', ...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.shap_wrappers (WARNING)>
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

### `KERNEL_SHAP_BACKGROUND_THRESHOLD`
```python
KERNEL_SHAP_BACKGROUND_THRESHOLD: int = 300
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### `DISTRIBUTED_OPTS`
```python
DISTRIBUTED_OPTS: dict = {'n_cpus': None, 'batch_size': 1}
```

### `TREE_SHAP_BACKGROUND_SUPPORTED_SIZE`
```python
TREE_SHAP_BACKGROUND_SUPPORTED_SIZE: int = 100
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### `TREE_SHAP_BACKGROUND_WARNING_THRESHOLD`
```python
TREE_SHAP_BACKGROUND_WARNING_THRESHOLD: int = 1000
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### `TREE_SHAP_MODEL_OUTPUT`
```python
TREE_SHAP_MODEL_OUTPUT: list = ['raw', 'probability', 'probability_doubled', 'log_loss']
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

## `KernelExplainerWrapper`

_Inherits from:_ `KernelExplainer`, `Explainer`, `Serializable`

A wrapper around `shap.KernelExplainer` that supports:

- fixing the seed when instantiating the KernelExplainer in a separate process.

- passing a batch index to the explainer so that a parallel explainer pool can return batches in         arbitrary order.

### Constructor

```python
KernelExplainerWrapper(self, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `*args,` | `**kwargs` |  | Arguments and keyword arguments for `shap.KernelExplainer` constructor. |

### Methods

#### `get_explanation`

```python
get_explanation(X: Union[Tuple[int, numpy.ndarray], numpy.ndarray], kwargs) -> Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[Tuple[int, numpy.ndarray], numpy.ndarray]` |  | When called from a distributed context, it is a tuple containing a batch index and a batch to be explained. Otherwise, it is an array of instances to be explained. |
| `**kwargs` |  |  | `shap.KernelExplainer.shap_values` kwarg values. |

**Returns**
- Type: `Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]`

#### `return_attribute`

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

## `KernelShap`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
KernelShap(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], link: str = 'identity', feature_names: Union[List[str], Tuple[str], NoneType] = None, categorical_names: Optional[Dict[int, List[str]]] = None, task: str = 'classification', seed: Optional[int] = None, distributed_opts: Optional[Dict] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes as an input a `samples x features` array and outputs a `samples x n_outputs` model outputs. The `n_outputs` should represent model output in margin space. If the model outputs probabilities, then the link should be set to ``'logit'`` to ensure correct force plots. |
| `link` | `str` | `'identity'` | Valid values are ``'identity'`` or ``'logit'``. A generalized linear model link to connect the feature importance values to the model output. Since the feature importance values, :math:`\phi`, sum up to the model output, it often makes sense to connect them to the ouput with a link function where :math:`link(output - expected\_value) = sum(\phi)`. Therefore, for a model which outputs probabilities, ``link='logit'`` makes the feature effects have log-odds (evidence) units and ``link='identity'`` means that the feature effects have probability units. Please see this `example`_ for an in-depth discussion about the semantics of explaining the model in the probability or margin space. .. _example: https://github.com/slundberg/shap/blob/master/notebooks/tabular_examples/model_agnostic/Squashing%20Effect.ipynb |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` | Used to infer group names when categorical data is treated by grouping and `group_names` input to `fit` is not specified, assuming it has the same length as the `groups` argument of `fit` method. It is also used to compute the `names` field, which appears as a key in each of the values of `explanation.data['raw']['importances']`. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Keys are feature column indices in the `background_data` matrix (see `fit`). Each value contains strings with the names of the categories for the feature. Used to select the method for background data summarisation (if specified, subsampling is performed as opposed to k-means clustering). In the future it may be used for visualisation. |
| `task` | `str` | `'classification'` | Can have values ``'classification'`` and ``'regression'``. It is only used to set the contents of `explanation.data['raw']['prediction']` |
| `seed` | `Optional[int]` | `None` | Fixes the random number stream, which influences which subsets are sampled during shap value estimation. |
| `distributed_opts` | `Optional[Dict]` | `None` | A dictionary that controls the algorithm distributed execution. See :py:data:`alibi.explainers.shap_wrappers.DISTRIBUTED_OPTS` documentation for details. |

### Methods

#### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix]` |  | Instances to be explained. |
| `summarise_result` | `bool` | `False` | Specifies whether the shap values corresponding to dimensions of encoded categorical variables should be summed so that a single shap value is returned for each categorical variable. Both the start indices of the categorical variables (`cat_vars_start_idx`) and the encoding dimensions (`cat_vars_enc_dim`) have to be specified |
| `cat_vars_start_idx` | `Optional[Sequence[int]]` | `None` | The start indices of the categorical variables. If specified, `cat_vars_enc_dim` should also be specified. |
| `cat_vars_enc_dim` | `Optional[Sequence[int]]` | `None` | The length of the encoding dimension for each categorical variable. If specified `cat_vars_start_idx` should also be specified. |
| `**kwargs` |  |  | Keyword arguments specifying explain behaviour. Valid arguments are: - `nsamples` - controls the number of predictor calls and therefore runtime. - `l1_reg` - the algorithm is exponential in the feature dimension. If set to `auto` the algorithm will                 first run a feature selection algorithm to select the top features, provided the fraction of sampled                 sets of missing features is less than 0.2 from the number of total subsets. The Akaike Information                 Criterion is used in this case. See our examples for more details about available settings for this                 parameter. Note that by first running a feature selection step, the shapley values of the remainder of                 the features will be different to those estimated from the entire set. For more details, please see the shap library `documentation`_ . .. _documentation: https://shap.readthedocs.io/en/stable/. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(background_data: Union[numpy.ndarray, scipy.sparse._matrix.spmatrix, pandas.core.frame.DataFrame, shap.utils._legacy.Data], summarise_background: Union[bool, str] = False, n_background_samples: int = 300, group_names: Union[List[str], Tuple[str], None] = None, groups: Optional[List[Union[Tuple[int], List[int]]]] = None, weights: Union[List[float], Tuple[float], numpy.ndarray, None] = None, kwargs) -> alibi.explainers.shap_wrappers.KernelShap
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, scipy.sparse._matrix.spmatrix, pandas.core.frame.DataFrame, shap.utils._legacy.Data]` |  | Data used to estimate feature contributions and baseline values for force plots. The rows of the background data should represent samples and the columns features. |
| `summarise_background` | `Union[bool, str]` | `False` | A large background dataset impacts the runtime and memory footprint of the algorithm. By setting this argument to ``True``, only `n_background_samples` from the provided data are selected. If group_names or groups arguments are specified, the algorithm assumes that the data contains categorical variables so the records are selected uniformly at random. Otherwise, `shap.kmeans` (a wrapper around `sklearn` k-means implementation) is used for selection. If set to ``'auto'``, a default of `KERNEL_SHAP_BACKGROUND_THRESHOLD` samples is selected. |
| `n_background_samples` | `int` | `300` | The number of samples to keep in the background dataset if ``summarise_background=True``. |
| `group_names` | `Union[List[str], Tuple[str], None]` | `None` | If specified, this array is used to treat groups of features as one during feature perturbation. This feature can be useful, for example, to treat encoded categorical variables as one and can result in computational savings (this may require adjusting the `nsamples` parameter). |
| `groups` | `Optional[List[Union[Tuple[int], List[int]]]]` | `None` | A list containing sub-lists specifying the indices of features belonging to the same group. |
| `weights` | `Union[List[float], Tuple[float], numpy.ndarray, None]` | `None` | A sequence or array of weights. This is used only if grouping is specified and assigns a weight to each point in the dataset. |
| `**kwargs` |  |  | Expected keyword arguments include `keep_index` (bool) and should be used if a data frame containing an index column is passed to the algorithm. |

**Returns**
- Type: `alibi.explainers.shap_wrappers.KernelShap`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New prediction function. |

**Returns**
- Type: `None`

## `TreeShap`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
TreeShap(self, predictor: Any, model_output: str = 'raw', feature_names: Union[List[str], Tuple[str], NoneType] = None, categorical_names: Optional[Dict[int, List[str]]] = None, task: str = 'classification', seed: Optional[int] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  | A fitted model to be explained. `XGBoost`, `LightGBM`, `CatBoost` and most tree-based `scikit-learn` models are supported. In the future, `Pyspark` could also be supported. Please open an issue if this is a use case for you. |
| `model_output` | `str` | `'raw'` | Supported values are: ``'raw'``, ``'probability'``, ``'probability_doubled'``, ``'log_loss'``: - ``'raw'`` - the raw model of the output, which varies by task, is explained. This option                 should always be used if the `fit` is called without arguments. It should also be set to compute                 shap interaction values. For regression models it is the standard output, for binary classification                 in `XGBoost` it is the log odds ratio. - ``'probability'`` - the probability output is explained. This option should only be used if `fit`                 was called with the `background_data` argument set. The effect of specifying this parameter is that                 the `shap` library will use this information to transform the shap values computed in margin space                 (aka using the raw output) to shap values that sum to the probability output by the model plus the                 model expected output probability. This requires knowledge of the type of output for `predictor`                 which is inferred by the `shap` library from the model type (e.g., most sklearn models with exception                 of `sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.RandomForestClassifier`,                 `sklearn.ensemble.ExtraTreesClassifier` output logits) or on the basis of the mapping implemented in                 the `shap.TreeEnsemble` constructor. Only trees that output log odds and probabilities are supported                 currently. - ``'probability_doubled'`` - used for binary classification problem in situations where the model                 outputs the logits/probabilities for the positive class but shap values for both outcomes are desired.                 This option should be used only if `fit` was called with the `background_data` argument set. In                 this case the expected value for the negative class is 1 - expected_value for positive class and                 the shap values for the negative class are the negative values of the positive class shap values.                 As before, the explanation happens in the margin space, and the shap values are subsequently adjusted.                 convert the model output to probabilities. The same considerations as for `probability` apply for this                 output type too. - ``'log_loss'`` - logarithmic loss is explained. This option shoud be used only if `fit` was called                 with the `background_data` argument set and requires specifying labels, `y`, when calling `explain`.                 If the objective is squared error, then the transformation :math:`(output - y)^2` is applied. For                 binary cross-entropy objective, the transformation :math:`log(1 + exp(output)) - y * output` with                  :math:`y \in \{0, 1\}`. Currently only binary cross-entropy and squared error losses can be explained. |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` | Used to compute the `names` field, which appears as a key in each of the values of the `importances` sub-field of the response `raw` field. |
| `categorical_names` | `Optional[Dict[int, List[str]]]` | `None` | Keys are feature column indices. Each value contains strings with the names of the categories for the feature. Used to select the method for background data summarisation (if specified, subsampling is performed as opposed to kmeans clustering). In the future it may be used for visualisation. |
| `task` | `str` | `'classification'` | Can have values ``'classification'`` and ``'regression'``. It is only used to set the contents of the `prediction` field in the `data['raw']` response field. |
| `seed` | `Optional[int]` | `None` |  |

### Methods

#### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, ForwardRef('catboost.Pool')], y: Optional[numpy.ndarray] = None, interactions: bool = False, approximate: bool = False, check_additivity: bool = True, tree_limit: Optional[int] = None, summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, ForwardRef('catboost.Pool')]` |  | Instances to be explained. |
| `y` | `Optional[numpy.ndarray]` | `None` | Labels corresponding to rows of `X`. Should be passed only if a background dataset was passed to the `fit` method. |
| `interactions` | `bool` | `False` | If ``True``, the shap value for every feature of every instance in `X` is decomposed into `X.shape[1] - 1` shap value interactions and one main effect. This is only supported if `fit` is called with `background_dataset=None`. |
| `approximate` | `bool` | `False` | If ``True``, an approximation to the shap values that does not account for feature order is computed. This was proposed by `Ando Sabaas`_ here . Check `this`_ resource for more details. This option is currently only supported for `xgboost` and `sklearn` models. .. _Ando Sabaas: https://github.com/andosa/treeinterpreter .. _this: https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-019-0138-9/MediaObjects/42256_2019_138_MOESM1_ESM.pdf |
| `check_additivity` | `bool` | `True` | If ``True``, output correctness is ensured if ``model_output='raw'`` has been passed to the constructor. |
| `tree_limit` | `Optional[int]` | `None` | Explain the output of a subset of the first `tree_limit` trees in an ensemble model. |
| `summarise_result` | `bool` | `False` | This should be set to ``True`` only when some of the columns in `X` represent encoded dimensions of a categorical variable and one single shap value per categorical variable is desired. Both `cat_vars_start_idx` and `cat_vars_enc_dim` should be specified as detailed below to allow this. |
| `cat_vars_start_idx` | `Optional[Sequence[int]]` | `None` | The start indices of the categorical variables. |
| `cat_vars_enc_dim` | `Optional[Sequence[int]]` | `None` | The length of the encoding dimension for each categorical variable. |

**Returns**
- Type: `Explanation`

#### `fit`

```python
fit(background_data: Union[numpy.ndarray, pandas.core.frame.DataFrame, None] = None, summarise_background: Union[bool, str] = False, n_background_samples: int = 1000, kwargs) -> alibi.explainers.shap_wrappers.TreeShap
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, None]` | `None` | Data used to estimate feature contributions and baseline values for force plots. The rows of the background data should represent samples and the columns features. |
| `summarise_background` | `Union[bool, str]` | `False` | A large background dataset may impact the runtime and memory footprint of the algorithm. By setting this argument to ``True``, only `n_background_samples` from the provided data are selected. If the `categorical_names` argument has been passed to the constructor, subsampling of the data is used. Otherwise, `shap.kmeans` (a wrapper around `sklearn.kmeans` implementation) is used for selection. If set to ``'auto'``, a default of `TREE_SHAP_BACKGROUND_WARNING_THRESHOLD` samples is selected. |
| `n_background_samples` | `int` | `1000` | The number of samples to keep in the background dataset if ``summarise_background=True``. |

**Returns**
- Type: `alibi.explainers.shap_wrappers.TreeShap`

#### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  | New prediction. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shap_values` | `List[numpy.ndarray]` |  | Each element corresponds to a `samples x features` array of shap values corresponding to each model output. |
| `feature_names` | `Union[List[str], Tuple[str], None]` | `None` | Each element is the name of the column with the corresponding index in each of the arrays in the `shap_values` list. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  | A two or three dimensional array to be reduced, as described above. |
| `start_idx` | `Sequence[int]` |  | The start indices of the columns to be summed. |
| `enc_feat_dim` | `Sequence[int]` |  | The number of columns to be summed, one for each start index. |
