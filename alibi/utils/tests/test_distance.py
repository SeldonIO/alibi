import numpy as np
from scipy.spatial.distance import cityblock
from itertools import product
import pytest
from alibi.utils.distance import abdm, cityblock_batch, mvdm

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


n_cat = [2, 3, 4]
n_labels = [2, 3]
n_items = [20, 50, 100]
cols = [1, 5]
tests = list(product(n_cat, n_labels, n_items, cols))
n_tests = len(tests)


@pytest.fixture
def cats_and_labels(request):
    cat, label, items, cols = tests[request.param]
    cats = np.random.randint(0, cat, items * cols).reshape(-1, cols)
    labels = np.random.randint(0, label, items).reshape(-1, 1)
    return cats, labels


@pytest.mark.parametrize('cats_and_labels', list(range(n_tests)), indirect=True)
def test_abdm_mvdm(cats_and_labels):
    X, y = cats_and_labels
    n_cols = X.shape[1]
    cat_vars = {i: None for i in range(n_cols)}
    if n_cols > 1:
        d_pair = abdm(X, cat_vars)
    else:
        d_pair = mvdm(X, y, cat_vars)
    assert list(d_pair.keys()) == list(cat_vars.keys())
    assert d_pair[0].shape == (cat_vars[0], cat_vars[0])
    assert d_pair[0].min() >= 0
