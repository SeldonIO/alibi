# `alibi.utils.approximation_methods`
## Constants
### `SUPPORTED_RIEMANN_METHODS`
```python
SUPPORTED_RIEMANN_METHODS: list = ['riemann_left', 'riemann_right', 'riemann_middle', 'riemann_trapezoid']
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `SUPPORTED_METHODS`
```python
SUPPORTED_METHODS: list = ['riemann_left', 'riemann_right', 'riemann_middle', 'riemann_trapezoid', 'gau...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

## `Riemann`

_Inherits from:_ `Enum`

An enumeration.

## Functions
### `approximation_parameters`

```python
approximation_parameters(method: str) -> Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]
```

Retrieves parameters for the input approximation `method`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `method` | `str` |  | The name of the approximation method. Currently supported only: ``'riemann_*'`` and ``'gausslegendre``'. Check :py:data:`alibi.utils.approximation_methods.SUPPORTED_RIEMANN_METHODS` for all ``'riemann_*'`` possible values. |

**Returns**
- Type: `Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]`

### `gauss_legendre_builders`

```python
gauss_legendre_builders() -> Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]
```

`np.polynomial.legendre` function helps to compute step sizes and alpha coefficients using gauss-legendre
quadrature rule. Since `numpy` returns the integration parameters in different scales we need to rescale them to
adjust to the desired scale.

Gauss Legendre quadrature rule for approximating the integrals was originally
proposed by [Xue Feng and her intern Hauroun Habeeb]
(https://research.fb.com/people/feng-xue/).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n` |  |  | The number of integration steps. |

**Returns**
- Type: `Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]`

### `riemann_builders`

```python
riemann_builders(method: alibi.utils.approximation_methods.Riemann = <Riemann.trapezoid: 4>) -> Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]
```

Step sizes are identical and alphas are scaled in [0, 1].

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `method` | `alibi.utils.approximation_methods.Riemann` | `<Riemann.trapezoid: 4>` | Riemann method: ``Riemann.left`` | ``Riemann.right`` | ``Riemann.middle`` | ``Riemann.trapezoid``. |
| `n` |  |  | The number of integration steps. |

**Returns**
- Type: `Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]`
