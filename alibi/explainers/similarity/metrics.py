import numpy as np
from typing import Union


def dot(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
    """
    Performs a dot product between the vector(s) in X and vector Y.

    Parameters
    ----------
    x:
        Matrix of vectors.
    y:
        Single vector

    Returns
    -------
        Dot product between the vector(s) in X and vector Y.
    """
    if len(x.shape) == 1:
        assert x.shape == y.shape, "The vector `X` and `Y` need to have the same dimensions."
    else:
        assert x.shape[1] == y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`"

    return np.dot(x, y).item()


def cos(x: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the cosine between the vector(s) in X and vector Y.

    Parameters
    ----------
    x:
        Matrix of vectors.
    y:
        Single vector
    eps:
        Numerical stability.

    Returns
    -------
        Cosine between the vector(s) in X and vector Y.
    """

    if len(x.shape) == 1:
        assert x.shape == y.shape, "The vectors `X` and `Y` need to have the same dimensions."
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
    else:
        assert x.shape[1] == y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`"
        denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y)

    return np.dot(x, y) / (denominator + eps)


def influence(x: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the influence of instances X to instances Y. This is an asymmetric kernel.

    Parameters
    ----------
    x:
        Matrix of vectors.
    y:
        Singe vector.
    eps:
        Numerical stability.

    Returns
    -------
        Influence asymmetric kernel value.
    """
    if len(x.shape) == 1:
        assert x.shape == y.shape, "The vectors `X` and `Y` need to have the same dimensions."
        denominator = np.linalg.norm(x) ** 2
    else:
        assert x.shape[1] == y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`."
        denominator = np.linalg.norm(x, axis=1) ** 2

    return np.dot(x, y) / (denominator + eps)
