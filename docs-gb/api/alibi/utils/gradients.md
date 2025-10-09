# `alibi.utils.gradients`
## Functions
### `num_grad_batch`

```python
num_grad_batch(func: Callable, X: numpy.ndarray, args: Tuple = (), eps: Union[float, numpy.ndarray] = 1e-08) -> numpy.ndarray
```

Calculate the numerical gradients of a vector-valued function (typically a prediction function in classification)

with respect to a batch of arrays `X`.

Parameters
----------
func
    Function to be differentiated.
X
    A batch of vectors at which to evaluate the gradient of the function.
args
    Any additional arguments to pass to the function.
eps
    Gradient step to use in the numerical calculation, can be a single `float` or one for each feature.

Returns
-------
An array of gradients at each point in the batch `X`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `func` | `Callable` |  |  |
| `X` | `numpy.ndarray` |  |  |
| `args` | `Tuple` | `()` |  |
| `eps` | `Union[float, numpy.ndarray]` | `1e-08` |  |

**Returns**
- Type: `numpy.ndarray`

### `perturb`

```python
perturb(X: numpy.ndarray, eps: Union[float, numpy.ndarray] = 1e-08, proba: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Apply perturbation to instance or prediction probabilities. Used for numerical calculation of gradients.

Parameters
----------
X
    Array to be perturbed.
eps
    Size of perturbation.
proba
    If ``True``, the net effect of the perturbation needs to be 0 to keep the sum of the probabilities equal to 1.

Returns
-------
Instances where a positive and negative perturbation is applied.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `eps` | `Union[float, numpy.ndarray]` | `1e-08` |  |
| `proba` | `bool` | `False` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
