# `alibi.explainers.backends.pytorch.cfrl_tabular`

This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class,
:py:class:`alibi.explainers.cfrl_tabular`, for the Pytorch backend.

## Functions
### `consistency_loss`

```python
consistency_loss(Z_cf_pred: torch.Tensor, Z_cf_tgt: torch.Tensor, kwargs)
```

Computes heterogeneous consistency loss.

Parameters
----------
Z_cf_pred
    Predicted counterfactual embedding.
Z_cf_tgt
    Counterfactual embedding target.

Returns
-------
Heterogeneous consistency loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `torch.Tensor` |  |  |
| `Z_cf_tgt` | `torch.Tensor` |  |  |

### `l0_ohe`

```python
l0_ohe(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none') -> torch.Tensor
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
| `input` | `torch.Tensor` |  |  |
| `target` | `torch.Tensor` |  |  |
| `reduction` | `str` | `'none'` |  |

**Returns**
- Type: `torch.Tensor`

### `l1_loss`

```python
l1_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none') -> torch.Tensor
```

Computes L1 loss.

Parameters
----------
input
    Input tensor.
target
    Target tensor.
reduction
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.

Returns
-------
L1 loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input` | `torch.Tensor` |  |  |
| `target` | `torch.Tensor` |  |  |
| `reduction` | `str` | `'none'` |  |

**Returns**
- Type: `torch.Tensor`

### `sample_differentiable`

```python
sample_differentiable(X_hat_split: List[torch.Tensor], category_map: Dict[int, List[str]]) -> List[torch.Tensor]
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
| `X_hat_split` | `List[torch.Tensor]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |

**Returns**
- Type: `List[torch.Tensor]`

### `sparsity_loss`

```python
sparsity_loss(X_hat_split: List[torch.Tensor], X_ohe: torch.Tensor, category_map: Dict[int, List[str]], weight_num: float = 1.0, weight_cat: float = 1.0)
```

Computes heterogeneous sparsity loss.

Parameters
----------
X_hat_split
    List of one-hot encoded reconstructed columns form the auto-encoder.
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
| `X_hat_split` | `List[torch.Tensor]` |  |  |
| `X_ohe` | `torch.Tensor` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `weight_num` | `float` | `1.0` |  |
| `weight_cat` | `float` | `1.0` |  |
