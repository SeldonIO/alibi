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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[Any]]` |  |  |
| `immutable_features` | `List[str]` |  |  |
| `conditional` | `bool` | `True` |  |

**Returns**
- Type: `numpy.ndarray`

### `generate_condition`

```python
generate_condition(X_ohe: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[str]], ranges: Dict[str, List[float]], immutable_features: List[str], conditional: bool = True) -> numpy.ndarray
```

Generates conditional vector.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `ranges` | `Dict[str, List[float]]` |  |  |
| `immutable_features` | `List[str]` |  |  |
| `conditional` | `bool` | `True` |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `numpy.ndarray` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `ranges` | `Dict[str, List[float]]` |  |  |
| `immutable_features` | `List[str]` |  |  |
| `conditional` | `bool` | `True` |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `condition` | `Dict[str, List[Union[float, str]]]` |  |  |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `immutable_features` | `Optional[List[str]]` | `None` |  |
| `diverse` |  | `False` |  |

**Returns**
- Type: `List[numpy.ndarray]`

### `get_conditional_dim`

```python
get_conditional_dim(feature_names: List[str], category_map: Dict[int, List[str]]) -> int
```

Computes the dimension of the conditional vector.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `condition` | `Dict[str, List[Union[float, str]]]` |  |  |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |
| `ranges` | `Optional[Dict[str, List[float]]]` | `None` |  |
| `immutable_features` | `Optional[List[str]]` | `None` |  |
| `diverse` |  | `False` |  |

**Returns**
- Type: `numpy.ndarray`

### `get_he_preprocessor`

```python
get_he_preprocessor(X: numpy.ndarray, feature_names: List[str], category_map: Dict[int, List[str]], feature_types: Optional[Dict[str, type]] = None) -> Tuple[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]]
```

Heterogeneous dataset preprocessor. The numerical features are standardized and the categorical features

are one-hot encoded.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `feature_types` | `Optional[Dict[str, type]]` | `None` |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `condition` | `Dict[str, List[Union[float, str]]]` |  |  |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |
| `ranges` | `Optional[Dict[str, List[float]]]` | `None` |  |
| `immutable_features` | `Optional[List[str]]` | `None` |  |
| `diverse` |  | `False` |  |

**Returns**
- Type: `List[numpy.ndarray]`

### `get_statistics`

```python
get_statistics(X: numpy.ndarray, preprocessor: Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], category_map: Dict[int, List[str]]) -> Dict[int, Dict[str, float]]
```

Computes statistics.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `preprocessor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

**Returns**
- Type: `Dict[int, Dict[str, float]]`

### `sample`

```python
sample(X_hat_split: List[numpy.ndarray], X_ohe: numpy.ndarray, C: Optional[numpy.ndarray], category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]) -> List[numpy.ndarray]
```

Samples an instance from the given reconstruction according to the conditional vector and

the dictionary of statistics.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[numpy.ndarray]` |  |  |
| `X_ohe` | `numpy.ndarray` |  |  |
| `C` | `Optional[numpy.ndarray]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |

**Returns**
- Type: `List[numpy.ndarray]`

### `sample_categorical`

```python
sample_categorical(X_hat_cat_split: List[numpy.ndarray], C_cat_split: Optional[List[numpy.ndarray]]) -> List[numpy.ndarray]
```

Samples categorical features according to the conditional vector. This method sample conditional according to

the masking vector the most probable outcome.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_cat_split` | `List[numpy.ndarray]` |  |  |
| `C_cat_split` | `Optional[List[numpy.ndarray]]` |  |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_num_split` | `List[numpy.ndarray]` |  |  |
| `X_ohe_num_split` | `List[numpy.ndarray]` |  |  |
| `C_num_split` | `Optional[List[numpy.ndarray]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ohe` | `Union[np.ndarray, torch.Tensor, tf.Tensor]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

**Returns**
- Type: `Tuple[List[Any], List[Any]]`
