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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[torch.Tensor]` |  |  |
| `X_ohe` | `torch.Tensor` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `weight_num` | `float` | `1.0` |  |
| `weight_cat` | `float` | `1.0` |  |
