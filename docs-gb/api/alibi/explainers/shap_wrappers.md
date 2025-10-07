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
### Methods

### `get_explanation`

```python
get_explanation(X: Union[Tuple[int, numpy.ndarray], numpy.ndarray], kwargs) -> Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]
```

Wrapper around `shap.KernelExplainer.shap_values` that allows calling the method with a tuple containing a

batch index and a batch of instances.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[Tuple[int, numpy.ndarray], numpy.ndarray]` |  |  |

**Returns**
- Type: `Union[Tuple[int, numpy.ndarray], Tuple[int, List[numpy.ndarray]], numpy.ndarray, List[numpy.ndarray]]`

### `return_attribute`

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

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Constructor

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

### Methods

### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> alibi.api.interfaces.Explanation
```

Explains the instances in the array `X`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix]` |  |  |
| `summarise_result` | `bool` | `False` |  |
| `cat_vars_start_idx` | `Optional[Sequence[int]]` | `None` |  |
| `cat_vars_enc_dim` | `Optional[Sequence[int]]` | `None` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

### `fit`

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, scipy.sparse._matrix.spmatrix, pandas.core.frame.DataFrame, shap.utils._legacy.Data]` |  |  |
| `summarise_background` | `Union[bool, str]` | `False` |  |
| `n_background_samples` | `int` | `300` |  |
| `group_names` | `Union[List[str], Tuple[str], None]` | `None` |  |
| `groups` | `Optional[List[Union[Tuple[int], List[int]]]]` | `None` |  |
| `weights` | `Union[List[float], Tuple[float], numpy.ndarray, None]` | `None` |  |

**Returns**
- Type: `alibi.explainers.shap_wrappers.KernelShap`

### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the prediction function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

## `TreeShap`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Constructor

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

### Methods

### `explain`

```python
explain(X: Union[numpy.ndarray, pandas.core.frame.DataFrame, ForwardRef('catboost.Pool')], y: Optional[numpy.ndarray] = None, interactions: bool = False, approximate: bool = False, check_additivity: bool = True, tree_limit: Optional[int] = None, summarise_result: bool = False, cat_vars_start_idx: Optional[Sequence[int]] = None, cat_vars_enc_dim: Optional[Sequence[int]] = None, kwargs) -> Explanation
```

Explains the instances in `X`. `y` should be passed if the model loss function is to be explained,

which can be useful in order to understand how various features affect model performance over
time. This is only possible if the explainer has been fitted with a background dataset and
requires setting `model_output='log_loss'`.

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

**Returns**
- Type: `Explanation`

### `fit`

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `background_data` | `Union[numpy.ndarray, pandas.core.frame.DataFrame, None]` | `None` |  |
| `summarise_background` | `Union[bool, str]` | `False` |  |
| `n_background_samples` | `int` | `1000` |  |

**Returns**
- Type: `alibi.explainers.shap_wrappers.TreeShap`

### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

Resets the predictor.

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `values` | `numpy.ndarray` |  |  |
| `start_idx` | `Sequence[int]` |  |  |
| `enc_feat_dim` | `Sequence[int]` |  |  |
