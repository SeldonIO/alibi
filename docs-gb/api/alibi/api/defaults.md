# `alibi.api.defaults`

This module defines the default metadata and data dictionaries for each explanation method.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.

## Constants
### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_ANCHOR`
```python
DEFAULT_DATA_ANCHOR = {'anchor': [], 'precision': None, 'coverage': None, 'raw': None}
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_ANCHOR_IMG`
```python
DEFAULT_DATA_ANCHOR_IMG = {'anchor': [], 'segments': None, 'precision': None, 'coverage': None, 'raw': ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_CEM`
```python
DEFAULT_META_CEM = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_CEM`
```python
DEFAULT_DATA_CEM = {'PN': None, 'PP': None, 'PN_pred': None, 'PP_pred': None, 'grads_graph': Non...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_CF`
```python
DEFAULT_META_CF = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_CF`
```python
DEFAULT_DATA_CF = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'success': None}
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_CFP`
```python
DEFAULT_META_CFP = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_CFP`
```python
DEFAULT_DATA_CFP = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'id_proto': N...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `KERNEL_SHAP_PARAMS`
```python
KERNEL_SHAP_PARAMS: list = ['link', 'group_names', 'grouped', 'groups', 'weights', 'summarise_background...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `DEFAULT_META_KERNEL_SHAP`
```python
DEFAULT_META_KERNEL_SHAP = {'name': None, 'type': ['blackbox'], 'task': None, 'explanations': ['local', ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_KERNEL_SHAP`
```python
DEFAULT_DATA_KERNEL_SHAP = {'shap_values': [], 'expected_value': [], 'categorical_names': {}, 'feature_n...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_ALE`
```python
DEFAULT_META_ALE = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_ALE`
```python
DEFAULT_DATA_ALE = {'ale_values': [], 'constant_value': None, 'ale0': [], 'feature_values': [], ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `TREE_SHAP_PARAMS`
```python
TREE_SHAP_PARAMS: list = ['model_output', 'summarise_background', 'summarise_result', 'approximate', '...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `DEFAULT_META_TREE_SHAP`
```python
DEFAULT_META_TREE_SHAP = {'name': None, 'type': ['whitebox'], 'task': None, 'explanations': ['local', ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_TREE_SHAP`
```python
DEFAULT_DATA_TREE_SHAP = {'shap_values': [], 'shap_interaction_values': [], 'expected_value': [], 'cat...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_INTGRAD`
```python
DEFAULT_META_INTGRAD = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_INTGRAD`
```python
DEFAULT_DATA_INTGRAD = {'attributions': None, 'X': None, 'forward_kwargs': None, 'baselines': None, ...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_CFRL`
```python
DEFAULT_META_CFRL = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_CFRL`
```python
DEFAULT_DATA_CFRL = {'orig': None, 'cf': None, 'target': None, 'condition': None}
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_SIM`
```python
DEFAULT_META_SIM = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_SIM`
```python
DEFAULT_DATA_SIM = {'scores': None, 'ordered_indices': None, 'most_similar': None, 'least_simila...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_PROTOSELECT`
```python
DEFAULT_META_PROTOSELECT = {'name': None, 'type': ['data'], 'explanation': ['global'], 'params': {}, 've...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_PROTOSELECT`
```python
DEFAULT_DATA_PROTOSELECT = {'prototypes': None, 'prototype_indices': None, 'prototype_labels': None}
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_PD`
```python
DEFAULT_META_PD = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD = {'feature_deciles': None, 'pd_values': None, 'ice_values': None, 'feature_val...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_PDVARIANCE`
```python
DEFAULT_META_PDVARIANCE = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_PDVARIANCE`
```python
DEFAULT_DATA_PDVARIANCE = {'feature_deciles': None, 'pd_values': None, 'feature_values': None, 'feature...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_META_PERMUTATION_IMPORTANCE`
```python
DEFAULT_META_PERMUTATION_IMPORTANCE = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### `DEFAULT_DATA_PERMUTATION_IMPORTANCE`
```python
DEFAULT_DATA_PERMUTATION_IMPORTANCE = {'feature_names': None, 'metric_names': None, 'feature_importance': None}
```
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)
