from typing import Union

import numpy as np


def dot(X: np.ndarray, Y: np.ndarray) -> Union[float, np.ndarray]:
    """
    Performs a dot product between the vector(s) in X and vector Y.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Single vector

    Returns
    -------
        Dot product between the vector(s) in X and vector Y.
    """
    if len(X.shape) == 1:
        assert X.shape == Y.shape, "The vector `X` and `Y` need to have the same dimensions."
    else:
        assert X.shape[1] == Y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`"
    return np.dot(X, Y)


def cos(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the cosine between the vector(s) in X and vector Y.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Single vector
    eps
        Numerical stability.

    Returns
    -------
        Cosine between the vector(s) in X and vector Y.
    """

    if len(X.shape) == 1:
        assert X.shape == Y.shape, "The vectors `X` and `Y` need to have the same dimensions."
        denominator = np.linalg.norm(X) * np.linalg.norm(Y)
    else:
        assert X.shape[1] == Y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`"
        denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(Y)

    return np.dot(X, Y) / (denominator + eps)


def asym_dot(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the influence of instances `X` to instances `Y`. This is an asymmetric kernel.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Single vector.
    eps
        Numerical stability.

    Returns
    -------
        Influence asymmetric kernel value.
    """
    if len(X.shape) == 1:
        assert X.shape == Y.shape, "The vectors `X` and `Y` need to have the same dimensions."
        denominator = np.linalg.norm(X) ** 2
    else:
        assert X.shape[1] == Y.shape[0], "The second dimension of `X` need to be the same as the dimension of `Y`."
        denominator = np.linalg.norm(X, axis=1) ** 2

    return np.dot(X, Y) / (denominator + eps)
