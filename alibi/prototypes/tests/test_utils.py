import pytest
import numpy as np
from alibi.utils.distance import squared_pairwise_distance, batch_compute_kernel_matrix
from alibi.utils.kernel import GaussianRBF, GaussianRBFDistance, EuclideanDistance


@pytest.mark.parametrize('size', [100])
@pytest.mark.parametrize('ft_size', [2, 5, 10, 100])
def test_squared_pairwise_distance(size, ft_size):
    """
    Test vectorize implementation of the squared pairwise distance
    """
    np.random.seed(0)
    X = np.random.randn(size, ft_size)
    Y = np.random.randn(size, ft_size)

    # compute the naive way
    dist1 = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            dist1[i, j] = np.sum((X[i] - Y[j])**2)

    dist2 = squared_pairwise_distance(X, Y)
    assert np.allclose(dist1, dist2)


@pytest.mark.parametrize('size', [100])
@pytest.mark.parametrize('ft_size', [2, 5, 10, 100])
@pytest.mark.parametrize('sigma', [0.5])
def test_GaussianRBF(size, ft_size, sigma):
    """
    Test GaussianRBF kernel implementation
    """
    X = np.random.randn(size, ft_size)
    Y = np.random.randn(size, ft_size)

    kmatrix1 = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            kmatrix1[i, j] = np.exp(-np.linalg.norm(X[i] - Y[j])**2 / (2 * sigma**2))

    kernel = GaussianRBF(sigma=sigma)
    kmatrix2 = kernel(X, Y)
    assert np.allclose(kmatrix1, kmatrix2)


@pytest.mark.parametrize('size', [100])
@pytest.mark.parametrize('ft_size', [2, 5, 10, 100])
@pytest.mark.parametrize('sigma', [0.5])
def test_GaussianRBFDistance(size, ft_size, sigma):
    """
    Test GaussianRBF distance kernel implementation
    """
    X = np.random.randn(size, ft_size)
    Y = np.random.randn(size, ft_size)

    kmatrix1 = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            kmatrix1[i, j] = 1 - np.exp(-np.linalg.norm(X[i] - Y[j])**2 / (2 * sigma**2))

    kernel_distance = GaussianRBFDistance(sigma=sigma)
    kmatrix2 = kernel_distance(X, Y)
    assert np.allclose(kmatrix1, kmatrix2)


@pytest.mark.parametrize('size', [100])
@pytest.mark.parametrize('ft_size', [2, 5, 10, 100])
def test_EuclideanDistance(size, ft_size):
    """
    Test Euclidean distance kernel implementation
    """
    X = np.random.randn(size, ft_size)
    Y = np.random.randn(size, ft_size)

    kmatrix1 = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            kmatrix1[i, j] = np.linalg.norm(X[i] - Y[j])

    kernel_distance = EuclideanDistance()
    kmatrix2 = kernel_distance(X, Y)
    assert np.allclose(kmatrix1, kmatrix2)


@pytest.mark.parametrize('size', [100])
@pytest.mark.parametrize('ft_size', [2, 5, 10, 100])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 6, 8, 10])
@pytest.mark.parametrize('kernel', [EuclideanDistance()])
def test_batch_compute_kernel_matrix(size, ft_size, batch_size, kernel):
    """
    Test Euclidean distance kernel implementation
    """
    X = np.random.randn(size, ft_size)
    Y = np.random.randn(size, ft_size)

    kmatrix1 = kernel(X, Y)
    kmatrix2 = batch_compute_kernel_matrix(X, Y, kernel=kernel, batch_size=batch_size)
    assert np.allclose(kmatrix1, kmatrix2)
