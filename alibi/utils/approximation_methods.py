#!/usr/bin/env python3
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

SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]


def approximation_parameters(
    method: str,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    r"""Retrieves parameters for the input approximation `method`

        Args:
            method: The name of the approximation method. Currently only `riemann`
                    and gauss legendre are
    """
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))


def riemann_builders(
    method: Riemann = Riemann.trapezoid,
) -> Tuple[Callable[[int], List[float]], Callable[[int], List[float]]]:
    r"""Step sizes are identical and alphas are scaled in [0, 1]

        Args:

             n: The number of integration steps
             method: `left`, `right`, `middle` and `trapezoid` riemann

        Returns:
            2-element tuple of **step_sizes**, **alphas**:
            - **step_sizes** (*callable*):
                        `step_sizes` takes the number of steps as an
                        input argument and returns an array of steps sizes which
                        sum is smaller than or equal to one.

            - **alphas** (*callable*):
                        `alphas` takes the number of steps as an input argument
                        and returns the multipliers/coefficients for the inputs
                        of integrand in the range of [0, 1]

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
        # leades to faster approaximation. Test cases for small number of steps
        # do not make sense but for larger number of steps the approximation is
        # better therefore leaving this option available
        # if method == 'riemann_include_endpoints':
        #     return [i / (n - 1) for i in range(n)]

    return step_sizes, alphas


def gauss_legendre_builders() -> Tuple[
    Callable[[int], List[float]], Callable[[int], List[float]]
]:
    r""" Numpy's `np.polynomial.legendre` function helps to compute step sizes
    and alpha coefficients using gauss-legendre quadrature rule.
    Since numpy returns the integration parameters in different scales we need to
    rescale them to adjust to the desired scale.

    Gauss Legendre quadrature rule for approximating the integrals was originally
    proposed by [Xue Feng and her intern Hauroun Habeeb]
    (https://research.fb.com/people/feng-xue/).

    Args:

        n (int): The number of integration steps

    Returns:
        2-element tuple of **step_sizes**, **alphas**:
        - **step_sizes** (*callable*):
                    `step_sizes` takes the number of steps as an
                    input argument and returns an array of steps sizes which
                    sum is smaller than or equal to one.

        - **alphas** (*callable*):
                    `alphas` takes the number of steps as an input argument
                    and returns the multipliers/coefficients for the inputs
                    of integrand in the range of [0, 1]

    """

    def step_sizes(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from 2 to 1
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

    def alphas(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from [-1, 1] to [0, 1]
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))

    return step_sizes, alphas
