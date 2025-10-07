# `alibi.utils.kernel`
## `EuclideanDistance`

### Constructor

```python
EuclideanDistance(self) -> None
```

## `GaussianRBF`

### Constructor

```python
GaussianRBF(self, sigma: Union[float, numpy.ndarray, NoneType] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sigma` | `Union[float, numpy.ndarray, None]` | `None` | Kernel bandwidth. Not to be specified if being inferred or trained. Can pass multiple values to evaluate the kernel with and then average. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `sigma` | `numpy.ndarray` |  |

## `GaussianRBFDistance`

### Constructor

```python
GaussianRBFDistance(self, sigma: Union[float, numpy.ndarray, NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sigma` | `Union[float, numpy.ndarray, None]` | `None` | See :py:meth:`alibi.utils.kernel.GaussianRBF.__init__`. |
