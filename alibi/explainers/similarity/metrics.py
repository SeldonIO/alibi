from typing import Union

import numpy as np


def dot(X: np.ndarray, Y: np.ndarray) -> Union[float, np.ndarray]:
    """
    Performs a dot product between the vector(s) in X and vector Y. (:math:`X^T Y = \\sum_i X_i Y_i`). Each of `X` and
    `Y` should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.

    Returns
    -------
        Matrix of dot products between the vector(s) in X and vectors in Y.
    """
    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    return np.dot(X, Y.T)


def cos(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the cosine between the vector(s) in X and vector Y. (:math:`X^T Y/\\|X\\|\\|Y\\|`). Each of `X` and `Y`
    should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.
    eps
        Numerical stability.

    Returns
    -------
        Matrix of cosine similarities between the vector(s) in X and vectors in Y.
    """
    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    denominator = np.linalg.norm(X, axis=1)[:, None] @ np.linalg.norm(Y, axis=1)[None, :]
    return np.dot(X, Y.T) / (denominator + eps)


def asym_dot(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the influence of training instances `Y` to test instances `X`. This is an asymmetric kernel.
    (:math:`X^T Y/\\|Y\\|^2`). See the `paper <https://arxiv.org/abs/2102.05262>`_ for more details. Each of `X` and
    `Y` should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.
    eps
        Numerical stability.

    Returns
    -------
        Matrix of asymmetric dot product similarity values between the vector(s) in X and vectors in Y.
    """

    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    denominator = np.linalg.norm(Y, axis=1) ** 2
    return np.dot(X, Y.T) / (denominator + eps)[None, :]
