# `alibi.api.defaults`

This module defines the default metadata and data dictionaries for each explanation method.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.

## Constants
### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

### `DEFAULT_DATA_ANCHOR`
```python
DEFAULT_DATA_ANCHOR: dict = {'anchor': [], 'precision': None, 'coverage': None, 'raw': None}
```

### `DEFAULT_DATA_ANCHOR_IMG`
```python
DEFAULT_DATA_ANCHOR_IMG: dict = {'anchor': [], 'segments': None, 'precision': None, 'coverage': None, 'raw': ...
```

### `DEFAULT_META_CEM`
```python
DEFAULT_META_CEM: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `DEFAULT_DATA_CEM`
```python
DEFAULT_DATA_CEM: dict = {'PN': None, 'PP': None, 'PN_pred': None, 'PP_pred': None, 'grads_graph': Non...
```

### `DEFAULT_META_CF`
```python
DEFAULT_META_CF: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `DEFAULT_DATA_CF`
```python
DEFAULT_DATA_CF: dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'success': None}
```

### `DEFAULT_META_CFP`
```python
DEFAULT_META_CFP: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `DEFAULT_DATA_CFP`
```python
DEFAULT_DATA_CFP: dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'id_proto': N...
```

### `KERNEL_SHAP_PARAMS`
```python
KERNEL_SHAP_PARAMS: list = ['link', 'group_names', 'grouped', 'groups', 'weights', 'summarise_background...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `DEFAULT_META_KERNEL_SHAP`
```python
DEFAULT_META_KERNEL_SHAP: dict = {'name': None, 'type': ['blackbox'], 'task': None, 'explanations': ['local', ...
```

### `DEFAULT_DATA_KERNEL_SHAP`
```python
DEFAULT_DATA_KERNEL_SHAP: dict = {'shap_values': [], 'expected_value': [], 'categorical_names': {}, 'feature_n...
```

### `DEFAULT_META_ALE`
```python
DEFAULT_META_ALE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `DEFAULT_DATA_ALE`
```python
DEFAULT_DATA_ALE: dict = {'ale_values': [], 'constant_value': None, 'ale0': [], 'feature_values': [], ...
```

### `TREE_SHAP_PARAMS`
```python
TREE_SHAP_PARAMS: list = ['model_output', 'summarise_background', 'summarise_result', 'approximate', '...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `DEFAULT_META_TREE_SHAP`
```python
DEFAULT_META_TREE_SHAP: dict = {'name': None, 'type': ['whitebox'], 'task': None, 'explanations': ['local', ...
```

### `DEFAULT_DATA_TREE_SHAP`
```python
DEFAULT_DATA_TREE_SHAP: dict = {'shap_values': [], 'shap_interaction_values': [], 'expected_value': [], 'cat...
```

### `DEFAULT_META_INTGRAD`
```python
DEFAULT_META_INTGRAD: dict = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```

### `DEFAULT_DATA_INTGRAD`
```python
DEFAULT_DATA_INTGRAD: dict = {'attributions': None, 'X': None, 'forward_kwargs': None, 'baselines': None, ...
```

### `DEFAULT_META_CFRL`
```python
DEFAULT_META_CFRL: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

### `DEFAULT_DATA_CFRL`
```python
DEFAULT_DATA_CFRL: dict = {'orig': None, 'cf': None, 'target': None, 'condition': None}
```

### `DEFAULT_META_SIM`
```python
DEFAULT_META_SIM: dict = {'name': None, 'type': ['whitebox'], 'explanations': ['local'], 'params': {},...
```

### `DEFAULT_DATA_SIM`
```python
DEFAULT_DATA_SIM: dict = {'scores': None, 'ordered_indices': None, 'most_similar': None, 'least_simila...
```

### `DEFAULT_META_PROTOSELECT`
```python
DEFAULT_META_PROTOSELECT: dict = {'name': None, 'type': ['data'], 'explanation': ['global'], 'params': {}, 've...
```

### `DEFAULT_DATA_PROTOSELECT`
```python
DEFAULT_DATA_PROTOSELECT: dict = {'prototypes': None, 'prototype_indices': None, 'prototype_labels': None}
```

### `DEFAULT_META_PD`
```python
DEFAULT_META_PD: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD: dict = {'feature_deciles': None, 'pd_values': None, 'ice_values': None, 'feature_val...
```

### `DEFAULT_META_PDVARIANCE`
```python
DEFAULT_META_PDVARIANCE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `DEFAULT_DATA_PDVARIANCE`
```python
DEFAULT_DATA_PDVARIANCE: dict = {'feature_deciles': None, 'pd_values': None, 'feature_values': None, 'feature...
```

### `DEFAULT_META_PERMUTATION_IMPORTANCE`
```python
DEFAULT_META_PERMUTATION_IMPORTANCE: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['global'], 'params': {}...
```

### `DEFAULT_DATA_PERMUTATION_IMPORTANCE`
```python
DEFAULT_DATA_PERMUTATION_IMPORTANCE: dict = {'feature_names': None, 'metric_names': None, 'feature_importance': None}
```
