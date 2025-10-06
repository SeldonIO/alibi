# `alibi.utils.approximation_methods`
## Classes
### `Riemann` (_inherits from `Enum`)

An enumeration.

#### Constructor

```python
Riemann(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

## Functions
### `approximation_parameters`

```python
approximation_parameters(method: str) -> Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]
```

Retrieves parameters for the input approximation `method`.

Parameters
----------
method
    The name of the approximation method. Currently supported only: ``'riemann_*'`` and ``'gausslegendre``'.
    Check :py:data:`alibi.utils.approximation_methods.SUPPORTED_RIEMANN_METHODS` for all ``'riemann_*'`` possible
    values.

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

Parameters
----------
n
    The number of integration steps.

Returns
-------
2-element tuple consisting of

 - `step_sizes` : ``Callable`` - `step_sizes` takes the number of steps as an input argument and returns an      array of steps sizes which sum is smaller than or equal to one.

 - `alphas` : ``Callable`` - `alphas` takes the number of steps as an input argument and returns the      multipliers/coefficients for the inputs of integrand in the range of [0, 1].

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

Parameters
----------
n
    The number of integration steps.
method
    Riemann method: ``Riemann.left`` | ``Riemann.right`` | ``Riemann.middle`` | ``Riemann.trapezoid``.

Returns
-------
2-element tuple consisting of

 - `step_sizes` :  ``Callable`` - `step_sizes` takes the number of steps as an input argument and returns an      array of steps sizes which sum is smaller than or equal to one.

 - `alphas` : ``Callable`` - `alphas` takes the number of steps as an input argument and returns the      multipliers/coefficients for the inputs of integrand in the range of [0, 1].

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `method` | `alibi.utils.approximation_methods.Riemann` | `<Riemann.trapezoid: 4>` | Riemann method: ``Riemann.left`` | ``Riemann.right`` | ``Riemann.middle`` | ``Riemann.trapezoid``. |
| `n` |  |  | The number of integration steps. |

**Returns**
- Type: `Tuple[Callable[[.[<class 'int'>]], List[float]], Callable[[.[<class 'int'>]], List[float]]]`
