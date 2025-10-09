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
| `Z_cf_pred` | `tensorflow.python.framework.tensor.Tensor` |  | Counterfactual embedding prediction. |
| `Z_cf_tgt` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Counterfactual embedding target. |

### `l0_ohe`

```python
l0_ohe(input: tensorflow.python.framework.tensor.Tensor, target: tensorflow.python.framework.tensor.Tensor, reduction: str = 'none') -> tensorflow.python.framework.tensor.Tensor
```

Computes the L0 loss for a one-hot encoding representation.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `target` | `tensorflow.python.framework.tensor.Tensor` |  | Target tensor |
| `reduction` | `str` | `'none'` | Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `l1_loss`

```python
l1_loss(input: tensorflow.python.framework.tensor.Tensor, target = <class 'tensorflow.python.framework.tensor.Tensor'>, reduction: str = 'none') -> tensorflow.python.framework.tensor.Tensor
```

Computes the L1 loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input` | `tensorflow.python.framework.tensor.Tensor` |  | Input tensor. |
| `target` |  | `<class 'tensorflow.python.framework.tensor.Tensor'>` | Target tensor |
| `reduction` | `str` | `'none'` | Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `sample_differentiable`

```python
sample_differentiable(X_hat_split: List[tensorflow.python.framework.tensor.Tensor], category_map: Dict[int, List[str]]) -> List[tensorflow.python.framework.tensor.Tensor]
```

Samples differentiable reconstruction.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[tensorflow.python.framework.tensor.Tensor]` |  | List of reconstructed columns form the auto-encoder. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible values for an attribute. |

**Returns**
- Type: `List[tensorflow.python.framework.tensor.Tensor]`

### `sparsity_loss`

```python
sparsity_loss(X_hat_split: List[tensorflow.python.framework.tensor.Tensor], X_ohe: tensorflow.python.framework.tensor.Tensor, category_map: Dict[int, List[str]], weight_num: float = 1.0, weight_cat: float = 1.0)
```

Computes heterogeneous sparsity loss.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_hat_split` | `List[tensorflow.python.framework.tensor.Tensor]` |  | List of reconstructed columns form the auto-encoder. |
| `X_ohe` | `tensorflow.python.framework.tensor.Tensor` |  | One-hot encoded representation of the input. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible values for an attribute. |
| `weight_num` | `float` | `1.0` | Numerical loss weight. |
| `weight_cat` | `float` | `1.0` | Categorical loss weight. |
