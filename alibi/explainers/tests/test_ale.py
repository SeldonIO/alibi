import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
import pytest
from alibi.explainers.ale import ale_num, adaptive_grid, get_quantiles
from alibi.api.defaults import DEFAULT_DATA_ALE, DEFAULT_META_ALE


@pytest.mark.parametrize("min_bin_points", [1, 4, 10])
def test_ale_num_linear_regression(min_bin_points):
    """
    The slope of the ALE of linear regression should equal the learnt coefficients
    """
    X, y = load_boston(return_X_y=True)
    lr = LinearRegression().fit(X, y)
    for feature in range(X.shape[1]):
        q, ale, _ = ale_num(lr.predict, X, feature=feature, min_bin_points=min_bin_points)
        assert np.allclose((ale[-1] - ale[0]) / (X[:, feature].max() - X[:, feature].min()), lr.coef_[feature])


@pytest.mark.parametrize("min_bin_points", [1, 4, 10])
def test_ale_num_logistic_regression(min_bin_points):
    """
    The slope of the ALE curves performed in the logit space should equal the learnt coefficients.
    """
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(max_iter=200).fit(X, y)
    for feature in range(X.shape[1]):
        q, ale, _ = ale_num(lr.decision_function, X, feature=feature, min_bin_points=min_bin_points)
        alediff = ale[-1, :] - ale[0, :]
        xdiff = X[:, feature].max() - X[:, feature].min()
        assert np.allclose(alediff / xdiff, lr.coef_[:, feature])


@pytest.mark.parametrize('input_dim', (1, 10), ids='input_dim={}'.format)
@pytest.mark.parametrize('batch_size', (100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('num_points', (6, 11, 101), ids='num_points={}'.format)
def test_get_quantiles(input_dim, batch_size, num_points):
    X = np.random.rand(batch_size, input_dim)
    q = get_quantiles(X, num_points=num_points)
    assert q.shape == (num_points, input_dim)


@pytest.mark.parametrize('batch_size', (100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('min_bin_points', (1, 5, 10), ids='min_bin_points={}'.format)
def test_adaptive_grid(batch_size, min_bin_points):
    X = np.random.rand(batch_size, )
    q, num_points = adaptive_grid(X, min_bin_points=min_bin_points)

    # check that each bin has >= min_bin_points
    indices = np.searchsorted(q, X, side='left')  # assign points to bins
    indices[indices == 0] = 1  # zeroth bin should be empty
    interval_n = np.bincount(indices)  # count points
    assert np.all(interval_n[1:] > min_bin_points)


out_dim_out_type = [(1, 'continuous'), (3, 'proba')]


@pytest.mark.parametrize('input_dim', (1, 10), ids='input_dim={}'.format)
@pytest.mark.parametrize('batch_size', (10, 100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('mock_ale_explainer', out_dim_out_type, indirect=True, ids='out_dim, out_type={}'.format)
def test_explain(mock_ale_explainer, input_dim, batch_size):
    X = np.random.rand(batch_size, input_dim)
    explanation = mock_ale_explainer.explain(X)

    assert explanation.meta.keys() == DEFAULT_META_ALE.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ALE.keys()
