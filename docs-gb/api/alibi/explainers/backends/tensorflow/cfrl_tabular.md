# `alibi.explainers.backends.tensorflow.cfrl_tabular`

This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class (`cfrl_tabular`)
for the Tensorflow backend.

## Functions
### `consistency_loss`

```python
consistency_loss(Z_cf_pred: tensorflow.python.framework.tensor.Tensor, Z_cf_tgt: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], kwargs)
```

Computes heterogeneous consistency loss.

Parameters
----------
Z_cf_pred
        Counterfactual embedding prediction.
Z_cf_tgt
    Counterfactual embedding target.

Returns
-------
Heterogeneous consistency loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `Z_cf_tgt` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |
| `kwargs` |  |  |  |

### `l0_ohe`

```python
l0_ohe(input: tensorflow.python.framework.tensor.Tensor, target: tensorflow.python.framework.tensor.Tensor, reduction: str = 'none') -> tensorflow.python.framework.tensor.Tensor
```

Computes the L0 loss for a one-hot encoding representation.

Parameters
----------
input
    Input tensor.
target
    Target tensor
reduction
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.

Returns
-------
L0 loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `target` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `reduction` | `str` | `'none'` |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `l1_loss`

```python
l1_loss(input: tensorflow.python.framework.tensor.Tensor, target = <class 'tensorflow.python.framework.tensor.Tensor'>, reduction: str = 'none') -> tensorflow.python.framework.tensor.Tensor
```

Computes the L1 loss.

Parameters
----------
input
   Input tensor.
target
   Target tensor
reduction
   Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.

Returns
-------
L1 loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `target` |  | `<class 'tensorflow.python.framework.tensor.Tensor'>` |  |
| `reduction` | `str` | `'none'` |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `sample_differentiable`

```python
sample_differentiable(X_hat_split: List[tensorflow.python.framework.tensor.Tensor], category_map: Dict[int, List[str]]) -> List[tensorflow.python.framework.tensor.Tensor]
```

Samples differentiable reconstruction.

Parameters
----------
X_hat_split
    List of reconstructed columns form the auto-encoder.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
    values for an attribute.

Returns
-------
Differentiable reconstruction.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[tensorflow.python.framework.tensor.Tensor]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

**Returns**
- Type: `List[tensorflow.python.framework.tensor.Tensor]`

### `sparsity_loss`

```python
sparsity_loss(X_hat_split: List[tensorflow.python.framework.tensor.Tensor], X_ohe: tensorflow.python.framework.tensor.Tensor, category_map: Dict[int, List[str]], weight_num: float = 1.0, weight_cat: float = 1.0)
```

Computes heterogeneous sparsity loss.

Parameters
----------
X_hat_split
    List of reconstructed columns form the auto-encoder.
X_ohe
    One-hot encoded representation of the input.
category_map
    Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
    values for an attribute.
weight_num
    Numerical loss weight.
weight_cat
    Categorical loss weight.

Returns
-------
Heterogeneous sparsity loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[tensorflow.python.framework.tensor.Tensor]` |  |  |
| `X_ohe` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `weight_num` | `float` | `1.0` |  |
| `weight_cat` | `float` | `1.0` |  |
