# `alibi.explainers.similarity.metrics`
## Functions
### `asym_dot`

```python
asym_dot(X: numpy.ndarray, Y: numpy.ndarray, eps: float = 1e-07) -> Union[float, numpy.ndarray]
```

Computes the influence of training instances `Y` to test instances `X`. This is an asymmetric kernel.
(:math:`X^T Y/\|Y\|^2`). See the `paper <https://arxiv.org/abs/2102.05262>`_ for more details. Each of `X` and
`Y` should have a leading batch dimension of size at least 1.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Matrix of vectors. |
| `Y` | `numpy.ndarray` |  | Matrix of vectors. |
| `eps` | `float` | `1e-07` | Numerical stability. |

**Returns**
- Type: `Union[float, numpy.ndarray]`

### `cos`

```python
cos(X: numpy.ndarray, Y: numpy.ndarray, eps: float = 1e-07) -> Union[float, numpy.ndarray]
```

Computes the cosine between the vector(s) in X and vector Y. (:math:`X^T Y/\|X\|\|Y\|`). Each of `X` and `Y`
should have a leading batch dimension of size at least 1.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Matrix of vectors. |
| `Y` | `numpy.ndarray` |  | Matrix of vectors. |
| `eps` | `float` | `1e-07` | Numerical stability. |

**Returns**
- Type: `Union[float, numpy.ndarray]`

### `dot`

```python
dot(X: numpy.ndarray, Y: numpy.ndarray) -> Union[float, numpy.ndarray]
```

Performs a dot product between the vector(s) in X and vector Y. (:math:`X^T Y = \sum_i X_i Y_i`). Each of `X` and
`Y` should have a leading batch dimension of size at least 1.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Matrix of vectors. |
| `Y` | `numpy.ndarray` |  | Matrix of vectors. |

**Returns**
- Type: `Union[float, numpy.ndarray]`
