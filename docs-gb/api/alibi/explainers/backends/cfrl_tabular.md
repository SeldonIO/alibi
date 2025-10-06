# `alibi.explainers.backends.cfrl_tabular`

This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class,
:py:class:`alibi.explainers.cfrl_tabular`, that are common for both Tensorflow and Pytorch backends.

## Functions
### `apply_category_mapping`

```python
apply_category_mapping(X: numpy.ndarray, category_map: Dict[int, List[str]]) -> numpy.ndarray
```

Applies a category mapping for the categorical feature in the array. It transforms ints back to strings

to be readable.

Parameters
-----------
X
    Array containing the columns to be mapped.
category_map
    Dictionary of category mapping. Keys are columns index, and values are list of feature values.

Returns
-------
Transformed array.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array containing the columns to be mapped. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. Keys are columns index, and values are list of feature values. |

**Returns**
- Type: `numpy.ndarray`

### `generate_categorical_condition`

```python
generate_categorical_condition(X_ohe: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[Any]], immutable_features: List[str], conditional: bool = True) -> numpy.ndarray
```

Generates categorical features conditional vector. For a categorical feature of cardinality `K`, we condition the

subset of allowed feature through a binary mask of dimension `K`. When training the counterfactual generator,
the mask values are sampled from `Bern(0.5)`. For immutable features, only the original input feature value is
set to one in the binary mask. For example, the immutability of the ``'marital_status'`` having the current
value ``'married'`` is encoded through the binary sequence [1, 0, 0], given an ordering of the possible feature
values `[married, unmarried, divorced]`.

Parameters
----------
X_ohe
    One-hot encoding representation of the element(s) for which the conditional vector will be generated.
    The elements are required since some features can be immutable. In that case, the mask vector is the
    one-hot encoding itself for that particular feature.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values.
immutable_features
    List of immutable features.
conditional
    Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
    restrictions on the feature value.

Returns
-------
Conditional vector for categorical feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  | One-hot encoding representation of the element(s) for which the conditional vector will be generated. The elements are required since some features can be immutable. In that case, the mask vector is the one-hot encoding itself for that particular feature. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[Any]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. |
| `immutable_features` | `List[str]` |  | List of immutable features. |
| `conditional` | `bool` | `True` | Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any restrictions on the feature value. |

**Returns**
- Type: `numpy.ndarray`

### `generate_condition`

```python
generate_condition(X_ohe: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[str]], ranges: Dict[str, List[float]], immutable_features: List[str], conditional: bool = True) -> numpy.ndarray
```

Generates conditional vector.

Parameters
----------
X_ohe
    One-hot encoding representation of the element(s) for which the conditional vector will be generated.
    This method assumes that the input array, `X_ohe`, is has the first columns corresponding to the
    numerical features, and the rest are one-hot encodings of the categorical columns. The numerical and the
    categorical columns are ordered by the original column index( e.g., `numerical = (1, 4)`,
    `categorical=(0, 2, 3)`).
feature_names
    List of feature names.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values.
ranges
    Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
    negative and the second one positive.
immutable_features
    List of immutable map features.
conditional
    Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
    restrictions on the feature value.

Returns
-------
Conditional vector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  | One-hot encoding representation of the element(s) for which the conditional vector will be generated. This method assumes that the input array, `X_ohe`, is has the first columns corresponding to the numerical features, and the rest are one-hot encodings of the categorical columns. The numerical and the categorical columns are ordered by the original column index( e.g., `numerical = (1, 4)`, `categorical=(0, 2, 3)`). |
| `feature_names` | `List[str]` |  | List of feature names. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. |
| `ranges` | `Dict[str, List[float]]` |  | Dictionary of ranges for numerical features. Each value is a list containing two elements, first one negative and the second one positive. |
| `immutable_features` | `List[str]` |  | List of immutable map features. |
| `conditional` | `bool` | `True` | Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any restrictions on the feature value. |

**Returns**
- Type: `numpy.ndarray`

### `generate_numerical_condition`

```python
generate_numerical_condition(X_ohe: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[str]], ranges: Dict[str, List[float]], immutable_features: List[str], conditional: bool = True) -> numpy.ndarray
```

Generates numerical features conditional vector. For numerical features with a minimum value `a_min` and a

maximum value `a_max`, we include in the conditional vector the values `-p_min`, `p_max`, where `p_min, p_max`
are in [0, 1]. The range `[-p_min, p_max]` encodes a shift and scale-invariant representation of the interval
`[a - p_min * (a_max - a_min), a + p_max * (a_max - a_min)], where `a` is the original feature value. During
training, `p_min` and `p_max` are sampled from `Beta(2, 2)` for each unconstrained feature. Immutable features
can be encoded by `p_min = p_max = 0` or listed in `immutable_features` list. Features allowed to increase or
decrease only correspond to setting `p_min = 0` or `p_max = 0`, respectively. For example, allowing the ``'Age'``
feature to increase by up to 5 years is encoded by taking `p_min = 0`, `p_max=0.1`, assuming the minimum age of
10 and the maximum age of 60 years in the training set: `5 = 0.1 * (60 - 10)`.

Parameters
----------
X_ohe
    One-hot encoding representation of the element(s) for which the conditional vector will be generated.
    This argument is used to extract the number of conditional vector. The choice of `X_ohe` instead of a
    `size` argument is for consistency purposes with `categorical_cond` function.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values.
ranges:
    Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
    negative and the second one positive.
immutable_features
    Dictionary of immutable features. The keys are the column indexes and the values are booleans: ``True`` if
    the feature is immutable, ``False`` otherwise.
conditional
    Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
    restrictions on the feature value.

Returns
-------
Conditional vector for numerical features.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  | One-hot encoding representation of the element(s) for which the conditional vector will be generated. This argument is used to extract the number of conditional vector. The choice of `X_ohe` instead of a `size` argument is for consistency purposes with `categorical_cond` function. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. |
| `ranges` | `Dict[str, List[float]]` |  | Dictionary of ranges for numerical features. Each value is a list containing two elements, first one negative and the second one positive. |
| `immutable_features` | `List[str]` |  | Dictionary of immutable features. The keys are the column indexes and the values are booleans: ``True`` if the feature is immutable, ``False`` otherwise. |
| `conditional` | `bool` | `True` | Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any restrictions on the feature value. |

**Returns**
- Type: `numpy.ndarray`

### `get_categorical_conditional_vector`

```python
get_categorical_conditional_vector(X: numpy.ndarray, condition: Dict[str, List[Union[float, str]]], preprocessor: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], feature_names: List[str], category_map: Dict[int, List[str]], immutable_features: Optional[List[str]] = None, diverse = False) -> List[numpy.ndarray]
```

Generates a conditional vector. The condition is expressed a a delta change of the feature.

For categorical feature, if the ``'Occupation'`` can change to ``'Blue-Collar'`` or ``'White-Collar'``, the delta
change is ``['Blue-Collar', 'White-Collar']``. Note that the original value is optional as it is
included by default.

Parameters
----------
X
    Instances for which to generate the conditional vector in the original input format.
condition
    Dictionary of conditions per feature. For numerical features it expects a range that contains the original
    value. For categorical features it expects a list of feature values per features that includes the original
    value.
preprocessor
    Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
    into one-hot encoding representation. By convention, numerical features should be first, followed by the
    rest of categorical ones.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values.  This should be provided by the dataset.
immutable_features
    List of immutable features.
diverse
    Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
    a diverse set of counterfactuals for a given input instance.

Returns
-------
List of conditional vectors for each categorical feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances for which to generate the conditional vector in the original input format. |
| `condition` | `Dict[str, List[Union[float, str]]]` |  | Dictionary of conditions per feature. For numerical features it expects a range that contains the original value. For categorical features it expects a list of feature values per features that includes the original value. |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones into one-hot encoding representation. By convention, numerical features should be first, followed by the rest of categorical ones. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values.  This should be provided by the dataset. |
| `immutable_features` | `Optional[List[str]]` | `None` | List of immutable features. |
| `diverse` |  | `False` | Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate a diverse set of counterfactuals for a given input instance. |

**Returns**
- Type: `List[numpy.ndarray]`

### `get_conditional_dim`

```python
get_conditional_dim(feature_names: List[str], category_map: Dict[int, List[str]]) -> int
```

Computes the dimension of the conditional vector.

Parameters
----------
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values. This should be provided by the dataset.

Returns
-------
Dimension of the conditional vector

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. This should be provided by the dataset. |

**Returns**
- Type: `int`

### `get_conditional_vector`

```python
get_conditional_vector(X: numpy.ndarray, condition: Dict[str, List[Union[float, str]]], preprocessor: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], feature_names: List[str], category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]], ranges: Optional[Dict[str, List[float]]] = None, immutable_features: Optional[List[str]] = None, diverse = False) -> numpy.ndarray
```

Generates a conditional vector. The condition is expressed a a delta change of the feature.

For numerical features, if the ``'Age'`` feature is allowed to increase up to 10 more years, the delta change is
[0, 10].  If the ``'Hours per week'`` is allowed to decrease down to -5 and increases up to +10, then the
delta change is [-5, +10]. Note that the interval must go include 0.

For categorical feature, if the ``'Occupation'`` can change to ``'Blue-Collar'`` or ``'White-Collar'``,
the delta change is ``['Blue-Collar', 'White-Collar']``. Note that the original value is optional as it is
included by default.

Parameters
----------
X
    Instances for which to generate the conditional vector in the original input format.
condition
    Dictionary of conditions per feature. For numerical features it expects a range that contains the original
    value. For categorical features it expects a list of feature values per features that includes the original
    value.
preprocessor
    Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
    into one-hot encoding representation. By convention, numerical features should be first, followed by the
    rest of categorical ones.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values.  This should be provided by the dataset.
stats
    Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
    feature in the training set. Each key is an index of the column and each value is another dictionary
    containing ``'min'`` and ``'max'`` keys.
ranges
    Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
    negative and the second one positive.
immutable_features
    List of immutable features.
diverse
    Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
    a diverse set of counterfactuals for a given input instance.

Returns
-------
Conditional vector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances for which to generate the conditional vector in the original input format. |
| `condition` | `Dict[str, List[Union[float, str]]]` |  | Dictionary of conditions per feature. For numerical features it expects a range that contains the original value. For categorical features it expects a list of feature values per features that includes the original value. |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones into one-hot encoding representation. By convention, numerical features should be first, followed by the rest of categorical ones. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values.  This should be provided by the dataset. |
| `stats` | `Dict[int, Dict[str, float]]` |  | Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical feature in the training set. Each key is an index of the column and each value is another dictionary containing ``'min'`` and ``'max'`` keys. |
| `ranges` | `Optional[Dict[str, List[float]]]` | `None` | Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one negative and the second one positive. |
| `immutable_features` | `Optional[List[str]]` | `None` | List of immutable features. |
| `diverse` |  | `False` | Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate a diverse set of counterfactuals for a given input instance. |

**Returns**
- Type: `numpy.ndarray`

### `get_he_preprocessor`

```python
get_he_preprocessor(X: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[str]], feature_types: Optional[Dict[str, type]] = None) -> Tuple[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]
```

Heterogeneous dataset preprocessor. The numerical features are standardized and the categorical features

are one-hot encoded.

Parameters
----------
X
    Data to fit.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values. This should be provided by the dataset.
feature_types
    Dictionary of type for the numerical features.

Returns
-------
preprocessor
    Data preprocessor.
inv_preprocessor
    Inverse data preprocessor (e.g., `inv_preprocessor(preprocessor(x)) = x` )

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Data to fit. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. This should be provided by the dataset. |
| `feature_types` | `Optional[Dict[str, type]]` | `None` | Dictionary of type for the numerical features. |

**Returns**
- Type: `Tuple[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]`

### `get_numerical_conditional_vector`

```python
get_numerical_conditional_vector(X: numpy.ndarray, condition: Dict[str, List[Union[float, str]]], preprocessor: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], feature_names: List[str], category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]], ranges: Optional[Dict[str, List[float]]] = None, immutable_features: Optional[List[str]] = None, diverse = False) -> List[numpy.ndarray]
```

Generates a conditional vector. The condition is expressed a a delta change of the feature.

For numerical features, if the ``'Age'`` feature is allowed to increase up to 10 more years, the delta change is
[0, 10].  If the ``'Hours per week'`` is allowed to decrease down to -5 and increases up to +10, then the
delta change is [-5, +10]. Note that the interval must go include 0.

Parameters
----------
X
    Instances for which to generate the conditional vector in the original input format.
condition
    Dictionary of conditions per feature. For numerical features it expects a range that contains the original
    value. For categorical features it expects a list of feature values per features that includes the original
    value.
preprocessor
    Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
    into one-hot encoding representation. By convention, numerical features should be first, followed by the
    rest of categorical ones.
feature_names
    List of feature names. This should be provided by the dataset.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values. This should be provided by the dataset.
stats
    Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
    feature in the training set. Each key is an index of the column and each value is another dictionary
    containing ``'min'`` and ``'max'`` keys.
ranges
    Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
    negative and the second one positive.
immutable_features
    List of immutable features.
diverse
    Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
    a diverse set of counterfactuals for a given input instance.

Returns
-------
List of conditional vectors for each numerical feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances for which to generate the conditional vector in the original input format. |
| `condition` | `Dict[str, List[Union[float, str]]]` |  | Dictionary of conditions per feature. For numerical features it expects a range that contains the original value. For categorical features it expects a list of feature values per features that includes the original value. |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones into one-hot encoding representation. By convention, numerical features should be first, followed by the rest of categorical ones. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. This should be provided by the dataset. |
| `stats` | `Dict[int, Dict[str, float]]` |  | Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical feature in the training set. Each key is an index of the column and each value is another dictionary containing ``'min'`` and ``'max'`` keys. |
| `ranges` | `Optional[Dict[str, List[float]]]` | `None` | Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one negative and the second one positive. |
| `immutable_features` | `Optional[List[str]]` | `None` | List of immutable features. |
| `diverse` |  | `False` | Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate a diverse set of counterfactuals for a given input instance. |

**Returns**
- Type: `List[numpy.ndarray]`

### `get_statistics`

```python
get_statistics(X: numpy.ndarray, preprocessor: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], category_map: Dict[int, List[str]]) -> Dict[int, Dict[str, float]]
```

Computes statistics.

Parameters
----------
X
    Instances for which to compute statistic.
preprocessor
    Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
    into one-hot encoding representation. By convention, numerical features should be first, followed by the
    rest of categorical ones.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible feature values. This should be provided by the dataset.

Returns
-------
Dictionary of statistics. For each numerical column, the minimum and maximum value is returned.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances for which to compute statistic. |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones into one-hot encoding representation. By convention, numerical features should be first, followed by the rest of categorical ones. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. This should be provided by the dataset. |

**Returns**
- Type: `Dict[int, Dict[str, float]]`

### `sample`

```python
sample(X_hat_split: List[numpy.ndarray], X_ohe: numpy.ndarray, C: Optional[numpy.ndarray], category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]) -> List[numpy.ndarray]
```

Samples an instance from the given reconstruction according to the conditional vector and

the dictionary of statistics.

Parameters
----------
X_hat_split
    List of reconstructed columns from the auto-encoder. The categorical columns contain logits.
X_ohe
    One-hot encoded representation of the input.
C
    Conditional vector.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
    values for a feature.
stats
    Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
    feature in the training set. Each key is an index of the column and each value is another dictionary
    containing ``'min'`` and ``'max'`` keys.

Returns
-------
X_ohe_hat_split
    Most probable reconstruction sample according to the auto-encoder, sampled according to the conditional vector
    and the dictionary of statistics. This method assumes that the input array, `X_ohe` , has the first columns
    corresponding to the numerical features, and the rest are one-hot encodings of the categorical columns.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[numpy.ndarray]` |  | List of reconstructed columns from the auto-encoder. The categorical columns contain logits. |
| `X_ohe` | `numpy.ndarray` |  | One-hot encoded representation of the input. |
| `C` | `Optional[numpy.ndarray]` |  | Conditional vector. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible values for a feature. |
| `stats` | `Dict[int, Dict[str, float]]` |  | Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical feature in the training set. Each key is an index of the column and each value is another dictionary containing ``'min'`` and ``'max'`` keys. |

**Returns**
- Type: `List[numpy.ndarray]`

### `sample_categorical`

```python
sample_categorical(X_hat_cat_split: List[numpy.ndarray], C_cat_split: Optional[List[numpy.ndarray]]) -> List[numpy.ndarray]
```

Samples categorical features according to the conditional vector. This method sample conditional according to

the masking vector the most probable outcome.

Parameters
----------
X_hat_cat_split
    List of reconstructed categorical heads from the auto-encoder. The categorical columns contain logits.
C_cat_split
    List of conditional vector for categorical heads.

Returns
-------
X_ohe_hat_cat
    List of one-hot encoded vectors sampled according to the conditional vector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cat_split` | `List[numpy.ndarray]` |  | List of reconstructed categorical heads from the auto-encoder. The categorical columns contain logits. |
| `C_cat_split` | `Optional[List[numpy.ndarray]]` |  | List of conditional vector for categorical heads. |

**Returns**
- Type: `List[numpy.ndarray]`

### `sample_numerical`

```python
sample_numerical(X_hat_num_split: List[numpy.ndarray], X_ohe_num_split: List[numpy.ndarray], C_num_split: Optional[List[numpy.ndarray]], stats: Dict[int, Dict[str, float]]) -> List[numpy.ndarray]
```

Samples numerical features according to the conditional vector. This method clips the values between the

desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
the maximum values from train training datasets stored in the dictionary of statistics.

Parameters
----------
X_hat_num_split
    List of reconstructed numerical heads from the auto-encoder. This list should contain a single element
    as all the numerical features are part of a singe linear layer output.
X_ohe_num_split
    List of original numerical heads. The list should contain a single element as part of the convention
    mentioned in the description of `X_ohe_hat_num`.
C_num_split
    List of conditional vector for numerical heads. The list should contain a single element as part of the
    convention mentioned in the description of `X_ohe_hat_num`.
stats
    Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
    feature in the training set. Each key is an index of the column and each value is another dictionary
    containing ``'min'`` and ``'max'`` keys.

Returns
-------
X_ohe_hat_num
    List of clamped input vectors according to the conditional vectors and the dictionary of statistics.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_num_split` | `List[numpy.ndarray]` |  | List of reconstructed numerical heads from the auto-encoder. This list should contain a single element as all the numerical features are part of a singe linear layer output. |
| `X_ohe_num_split` | `List[numpy.ndarray]` |  | List of original numerical heads. The list should contain a single element as part of the convention mentioned in the description of `X_ohe_hat_num`. |
| `C_num_split` | `Optional[List[numpy.ndarray]]` |  | List of conditional vector for numerical heads. The list should contain a single element as part of the convention mentioned in the description of `X_ohe_hat_num`. |
| `stats` | `Dict[int, Dict[str, float]]` |  | Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical feature in the training set. Each key is an index of the column and each value is another dictionary containing ``'min'`` and ``'max'`` keys. |

**Returns**
- Type: `List[numpy.ndarray]`

### `split_ohe`

```python
split_ohe(X_ohe: Union[np.ndarray, torch.Tensor, tf.Tensor], category_map: Dict[int, List[str]]) -> Tuple[List[Any], List[Any]]
```

Splits a one-hot encoding array in a list of numerical heads and a list of categorical heads. Since by

convention the numerical heads are merged in a single head, if the function returns a list of numerical heads,
then the size of the list is 1.

Parameters
----------
X_ohe
    One-hot encoding representation. This can be any type of tensor: `np.ndarray`, `torch.Tensor`, `tf.Tensor`.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the
    possible values of a feature.

Returns
-------
X_ohe_num_split
    List of numerical heads. If different than ``None``, the list's size is 1.
X_ohe_cat_split
    List of categorical one-hot encoded heads.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `Union[np.ndarray, torch.Tensor, tf.Tensor]` |  | One-hot encoding representation. This can be any type of tensor: `np.ndarray`, `torch.Tensor`, `tf.Tensor`. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible values of a feature. |

**Returns**
- Type: `Tuple[List[Any], List[Any]]`
