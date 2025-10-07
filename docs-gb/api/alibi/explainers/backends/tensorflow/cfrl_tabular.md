# `alibi.explainers.backends.tensorflow.cfrl_tabular`

This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class (`cfrl_tabular`)
for the Tensorflow backend.

## Functions
### `consistency_loss`

```python
consistency_loss(Z_cf_pred: tensorflow.python.framework.tensor.Tensor, Z_cf_tgt: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], kwargs)
```

Computes heterogeneous consistency loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `Z_cf_pred` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `Z_cf_tgt` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |

### `l0_ohe`

```python
l0_ohe(input: tensorflow.python.framework.tensor.Tensor, target: tensorflow.python.framework.tensor.Tensor, reduction: str = 'none') -> tensorflow.python.framework.tensor.Tensor
```

Computes the L0 loss for a one-hot encoding representation.

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[tensorflow.python.framework.tensor.Tensor]` |  |  |
| `X_ohe` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `weight_num` | `float` | `1.0` |  |
| `weight_cat` | `float` | `1.0` |  |
