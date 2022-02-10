# This module is copied from the captum library at
# https://github.com/pytorch/captum/blob/master/captum/attr/_utils/approximation_methods.py

from typing import List, Callable, Tuple
import numpy as np
from enum import Enum


class Riemann(Enum):
    left = 1
    right = 2
    middle = 3
    trapezoid = 4


SUPPORTED_RIEMANN_METHODS = [
    "riemann_left",
    "riemann_right",
    "riemann_middle",
    "riemann_trapezoid",
]
"""
Riemann integration methods.
"""

SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]


def approximation_parameters(
        method: str,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    """
    Retrieves parameters for the input approximation `method`.

    Parameters
    ----------
    method
        The name of the approximation method. Currently supported only: ``'riemann_*'`` and ``'gausslegendre``'.
        Check :py:data:`alibi.utils.approximation_methods.SUPPORTED_RIEMANN_METHODS` for all ``'riemann_*'`` possible
        values.
    """
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))


def riemann_builders(
        method: Riemann = Riemann.trapezoid,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    """
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

     - `step_sizes` :  ``Callable`` - `step_sizes` takes the number of steps as an input argument and returns an \
     array of steps sizes which sum is smaller than or equal to one.

     - `alphas` : ``Callable`` - `alphas` takes the number of steps as an input argument and returns the \
     multipliers/coefficients for the inputs of integrand in the range of [0, 1].

    """

    def step_sizes(n: int) -> List[float]:
        assert n > 1, "The number of steps has to be larger than one"
        deltas = [1 / n] * n
        if method == Riemann.trapezoid:
            deltas[0] /= 2
            deltas[-1] /= 2
        return deltas

    def alphas(n: int) -> List[float]:
        assert n > 1, "The number of steps has to be larger than one"
        if method == Riemann.trapezoid:
            return list(np.linspace(0, 1, n))
        elif method == Riemann.left:
            return list(np.linspace(0, 1 - 1 / n, n))
        elif method == Riemann.middle:
            return list(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
        elif method == Riemann.right:
            return list(np.linspace(1 / n, 1, n))
        else:
            raise AssertionError("Provided Reimann approximation method is not valid.")
        # This is not a standard riemann method but in many cases it
        # leads to faster approximation. Test cases for small number of steps
        # do not make sense but for larger number of steps the approximation is
        # better therefore leaving this option available
        # if method == 'riemann_include_endpoints':
        #     return [i / (n - 1) for i in range(n)]

    return step_sizes, alphas


def gauss_legendre_builders() -> Tuple[
    Callable[[int], List[float]], Callable[[int], List[float]]
]:
    """
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

     - `step_sizes` : ``Callable`` - `step_sizes` takes the number of steps as an input argument and returns an \
     array of steps sizes which sum is smaller than or equal to one.

     - `alphas` : ``Callable`` - `alphas` takes the number of steps as an input argument and returns the \
     multipliers/coefficients for the inputs of integrand in the range of [0, 1].

    """

    def step_sizes(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from 2 to 1
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])  # type: ignore

    def alphas(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from [-1, 1] to [0, 1]
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))  # type: ignore

    return step_sizes, alphas
