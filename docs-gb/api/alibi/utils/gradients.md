# `alibi.utils.gradients`
## Functions
### `num_grad_batch`

```python
num_grad_batch(func: Callable, X: numpy.ndarray, args: Tuple = (), eps: Union[float, numpy.ndarray] = 1e-08) -> numpy.ndarray
```

Calculate the numerical gradients of a vector-valued function (typically a prediction function in classification)
with respect to a batch of arrays `X`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `func` | `Callable` |  | Function to be differentiated. |
| `X` | `numpy.ndarray` |  | A batch of vectors at which to evaluate the gradient of the function. |
| `args` | `Tuple` | `()` |  |
| `eps` | `Union[float, numpy.ndarray]` | `1e-08` |  |

**Returns**
- Type: `numpy.ndarray`

### `perturb`

```python
perturb(X: numpy.ndarray, eps: Union[float, numpy.ndarray] = 1e-08, proba: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Apply perturbation to instance or prediction probabilities. Used for numerical calculation of gradients.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array to be perturbed. |
| `eps` | `Union[float, numpy.ndarray]` | `1e-08` | Size of perturbation. |
| `proba` | `bool` | `False` | If ``True``, the net effect of the perturbation needs to be 0 to keep the sum of the probabilities equal to 1. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
