import numpy as np
from scipy.spatial.distance import cityblock
from itertools import product
import pytest
from alibi.utils.distance import cityblock_batch

dims = np.array([1, 10, 50])
shapes = list(product(dims, dims))
n_tests = len(dims) ** 2


@pytest.fixture
def random_matrix(request):
    shape = shapes[request.param]
    matrix = np.random.rand(*shape)
    return matrix


@pytest.mark.parametrize('random_matrix', list(range(n_tests)), indirect=True)
def test_cityblock_batch(random_matrix):
    X = random_matrix
    y = X[np.random.choice(X.shape[0])]

    batch_dists = cityblock_batch(X, y)
    single_dists = np.array([cityblock(x, y) for x in X]).reshape(X.shape[0], -1)

    assert np.allclose(batch_dists, single_dists)
