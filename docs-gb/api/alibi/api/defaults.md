# `alibi.api.defaults`

This module defines the default metadata and data dictionaries for each explanation method.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.

## Constants
### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR: dict = {'explanations': ['local'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
Anchors

### `DEFAULT_DATA_ANCHOR`
```python
DEFAULT_DATA_ANCHOR: dict = {'anchor': [], 'coverage': None, 'precision': None, 'raw': None}
```
### `DEFAULT_DATA_ANCHOR_IMG`
```python
DEFAULT_DATA_ANCHOR_IMG: dict = {'anchor': [], 'coverage': None, 'precision': None, 'raw': None, 'segments': None}
```
### `DEFAULT_META_CEM`
```python
DEFAULT_META_CEM: dict = { 'explanations': ['local'],
  'name': None,
  'params': {},
  'type': ['blackbox', 'tensorflow', 'keras'],
  'version': None}
```
CEM

### `DEFAULT_DATA_CEM`
```python
DEFAULT_DATA_CEM: dict = { 'PN': None,
  'PN_pred': None,
  'PP': None,
  'PP_pred': None,
  'X': None,
  'X_pred': None,
  'grads_graph': None,
  'grads_num': None}
```
### `DEFAULT_META_CF`
```python
DEFAULT_META_CF: dict = { 'explanations': ['local'],
  'name': None,
  'params': {},
  'type': ['blackbox', 'tensorflow', 'keras'],
  'version': None}
```
Counterfactuals

### `DEFAULT_DATA_CF`
```python
DEFAULT_DATA_CF: dict = {'all': [], 'cf': None, 'orig_class': None, 'orig_proba': None, 'success': None}
```
### `DEFAULT_META_CFP`
```python
DEFAULT_META_CFP: dict = { 'explanations': ['local'],
  'name': None,
  'params': {},
  'type': ['blackbox', 'tensorflow', 'keras'],
  'version': None}
```
CFProto

### `DEFAULT_DATA_CFP`
```python
DEFAULT_DATA_CFP: dict = {'all': [], 'cf': None, 'id_proto': None, 'orig_class': None, 'orig_proba': None}
```
### `KERNEL_SHAP_PARAMS`
```python
KERNEL_SHAP_PARAMS: list = [ 'link',
  'group_names',
  'grouped',
  'groups',
  'weights',
  'summarise_background',
  'summarise_result',
  'transpose',
  'kwargs']
```
KernelSHAP

### `DEFAULT_META_KERNEL_SHAP`
```python
DEFAULT_META_KERNEL_SHAP: dict = { 'explanations': ['local', 'global'],
  'name': None,
  'params': { 'group_names': None,
              'grouped': None,
              'groups': None,
              'kwargs': None,
              'link': None,
              'summarise_background': None,
              'summarise_result': None,
              'transpose': None,
              'weights': None},
  'task': None,
  'type': ['blackbox'],
  'version': None}
```
### `DEFAULT_DATA_KERNEL_SHAP`
```python
DEFAULT_DATA_KERNEL_SHAP: dict = { 'categorical_names': {},
  'expected_value': [],
  'feature_names': [],
  'raw': {'importances': {}, 'instances': None, 'prediction': None, 'raw_prediction': None},
  'shap_values': []}
```
### `DEFAULT_META_ALE`
```python
DEFAULT_META_ALE: dict = {'explanations': ['global'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
ALE

### `DEFAULT_DATA_ALE`
```python
DEFAULT_DATA_ALE: dict = { 'ale0': [],
  'ale_values': [],
  'constant_value': None,
  'feature_deciles': None,
  'feature_names': None,
  'feature_values': [],
  'target_names': None}
```
### `TREE_SHAP_PARAMS`
```python
TREE_SHAP_PARAMS: list = [ 'model_output',
  'summarise_background',
  'summarise_result',
  'approximate',
  'interactions',
  'explain_loss',
  'algorithm',
  'kwargs']
```
TreeShap

### `DEFAULT_META_TREE_SHAP`
```python
DEFAULT_META_TREE_SHAP: dict = { 'explanations': ['local', 'global'],
  'name': None,
  'params': { 'algorithm': None,
              'approximate': None,
              'explain_loss': None,
              'interactions': None,
              'kwargs': None,
              'model_output': None,
              'summarise_background': None,
              'summarise_result': None},
  'task': None,
  'type': ['whitebox'],
  'version': None}
```
### `DEFAULT_DATA_TREE_SHAP`
```python
DEFAULT_DATA_TREE_SHAP: dict = { 'categorical_names': {},
  'expected_value': [],
  'feature_names': [],
  'raw': { 'importances': {},
           'instances': None,
           'labels': None,
           'loss': None,
           'prediction': None,
           'raw_prediction': None},
  'shap_interaction_values': [],
  'shap_values': []}
```
### `DEFAULT_META_INTGRAD`
```python
DEFAULT_META_INTGRAD: dict = {'explanations': ['local'], 'name': None, 'params': {}, 'type': ['whitebox'], 'version': None}
```
Integrated gradients

### `DEFAULT_DATA_INTGRAD`
```python
DEFAULT_DATA_INTGRAD: dict = { 'X': None,
  'attributions': None,
  'baselines': None,
  'deltas': None,
  'forward_kwargs': None,
  'predictions': None}
```
### `DEFAULT_META_CFRL`
```python
DEFAULT_META_CFRL: dict = {'explanations': ['local'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
### `DEFAULT_DATA_CFRL`
```python
DEFAULT_DATA_CFRL: dict = {'cf': None, 'condition': None, 'orig': None, 'target': None}
```
### `DEFAULT_META_SIM`
```python
DEFAULT_META_SIM: dict = {'explanations': ['local'], 'name': None, 'params': {}, 'type': ['whitebox'], 'version': None}
```
Similarity methods

### `DEFAULT_DATA_SIM`
```python
DEFAULT_DATA_SIM: dict = {'least_similar': None, 'most_similar': None, 'ordered_indices': None, 'scores': None}
```
### `DEFAULT_META_PROTOSELECT`
```python
DEFAULT_META_PROTOSELECT: dict = {'explanation': ['global'], 'name': None, 'params': {}, 'type': ['data'], 'version': None}
```
### `DEFAULT_DATA_PROTOSELECT`
```python
DEFAULT_DATA_PROTOSELECT: dict = {'prototype_indices': None, 'prototype_labels': None, 'prototypes': None}
```
### `DEFAULT_META_PD`
```python
DEFAULT_META_PD: dict = {'explanations': ['global'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
PartialDependence

### `DEFAULT_DATA_PD`
```python
DEFAULT_DATA_PD: dict = { 'feature_deciles': None,
  'feature_names': None,
  'feature_values': None,
  'ice_values': None,
  'pd_values': None}
```
### `DEFAULT_META_PDVARIANCE`
```python
DEFAULT_META_PDVARIANCE: dict = {'explanations': ['global'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
PartialDependenceVariance

### `DEFAULT_DATA_PDVARIANCE`
```python
DEFAULT_DATA_PDVARIANCE: dict = { 'conditional_importance': None,
  'conditional_importance_values': None,
  'feature_deciles': None,
  'feature_importance': None,
  'feature_interaction': None,
  'feature_names': None,
  'feature_values': None,
  'pd_values': None}
```
### `DEFAULT_META_PERMUTATION_IMPORTANCE`
```python
DEFAULT_META_PERMUTATION_IMPORTANCE: dict = {'explanations': ['global'], 'name': None, 'params': {}, 'type': ['blackbox'], 'version': None}
```
PermutationImportance

### `DEFAULT_DATA_PERMUTATION_IMPORTANCE`
```python
DEFAULT_DATA_PERMUTATION_IMPORTANCE: dict = {'feature_importance': None, 'feature_names': None, 'metric_names': None}
```
