import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
from numpy.testing import assert_allclose
from alibi.explainers.ale import ale_num, adaptive_grid, get_quantiles, minimum_satisfied
from alibi.api.defaults import DEFAULT_DATA_ALE, DEFAULT_META_ALE


@pytest.mark.parametrize('min_bin_points', [1, 4, 10])
@pytest.mark.parametrize('dataset', [lazy_fixture('boston_data')])
@pytest.mark.parametrize('lr_regressor',
                         [lazy_fixture('boston_data')],
                         indirect=True,
                         ids='reg=lr_{}'.format)
def test_ale_num_linear_regression(min_bin_points, lr_regressor, dataset):
    """
    The slope of the ALE of linear regression should equal the learnt coefficients
    """
    lr, _ = lr_regressor
    X = dataset['X_train']

    for feature in range(X.shape[1]):
        q, ale, _ = ale_num(lr.predict, X, feature=feature, min_bin_points=min_bin_points)
        alediff = ale[-1] - ale[0]
        xdiff = X[:, feature].max() - X[:, feature].min()
        assert_allclose(alediff / xdiff, lr.coef_[feature])


@pytest.mark.parametrize('min_bin_points', [1, 4, 10])
@pytest.mark.parametrize('dataset', [lazy_fixture('iris_data')])
@pytest.mark.parametrize('lr_classifier',
                         [lazy_fixture('iris_data')],
                         indirect=True,
                         ids='clf=lr_{}'.format)
def test_ale_num_logistic_regression(min_bin_points, lr_classifier, dataset):
    """
    The slope of the ALE curves performed in the logit space should equal the learnt coefficients.
    """
    lr, _ = lr_classifier
    X = dataset['X_train']

    for feature in range(X.shape[1]):
        q, ale, _ = ale_num(lr.decision_function, X, feature=feature, min_bin_points=min_bin_points)
        alediff = ale[-1, :] - ale[0, :]
        xdiff = X[:, feature].max() - X[:, feature].min()
        assert_allclose(alediff / xdiff, lr.coef_[:, feature])


@pytest.mark.parametrize('input_dim', (1, 10), ids='input_dim={}'.format)
@pytest.mark.parametrize('batch_size', (100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('num_points', (6, 11, 101), ids='num_points={}'.format)
def test_get_quantiles(input_dim, batch_size, num_points):
    X = np.random.rand(batch_size, input_dim)
    q = get_quantiles(X, num_quantiles=num_points)
    assert q.shape == (num_points, input_dim)


@pytest.mark.parametrize('batch_size', (100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('min_bin_points', (1, 5, 10), ids='min_bin_points={}'.format)
def test_adaptive_grid(batch_size, min_bin_points):
    X = np.random.rand(batch_size, )
    q, num_points = adaptive_grid(X, min_bin_points=min_bin_points)

    # check that each bin has >= min_bin_points
    assert minimum_satisfied(X, min_bin_points, num_points)


out_dim_out_type = [(1, 'continuous'), (3, 'proba')]
features = [None, [0], [3, 5, 7]]


def uncollect_if_n_features_more_than_input_dim(**kwargs):
    features = kwargs['features']
    if features:
        n_features = len(features)
    else:
        n_features = kwargs['input_dim']

    return n_features > kwargs['input_dim']


@pytest.mark.uncollect_if(func=uncollect_if_n_features_more_than_input_dim)
@pytest.mark.parametrize('features', features, ids='features={}'.format)
@pytest.mark.parametrize('input_dim', (1, 10), ids='input_dim={}'.format)
@pytest.mark.parametrize('batch_size', (10, 100, 1000), ids='batch_size={}'.format)
@pytest.mark.parametrize('mock_ale_explainer', out_dim_out_type, indirect=True, ids='out_dim, out_type={}'.format)
def test_explain(mock_ale_explainer, features, input_dim, batch_size):
    out_dim = mock_ale_explainer.predictor.out_dim
    X = np.random.rand(batch_size, input_dim)

    if features:
        n_features = len(features)
    else:
        n_features = input_dim

    exp = mock_ale_explainer.explain(X, features=features)

    # check that the length of all relevant attributes is the same as the number of features explained
    assert all(len(attr) == n_features for attr in (exp.ale_values, exp.feature_values,
                                                    exp.feature_names, exp.feature_deciles,
                                                    exp.ale0))

    assert len(exp.target_names) == out_dim

    for alev, featv in zip(exp.ale_values, exp.feature_values):
        assert alev.shape == (featv.shape[0], out_dim)

    assert isinstance(exp.constant_value, float)

    for a0 in exp.ale0:
        assert a0.shape == (out_dim,)

    assert exp.meta.keys() == DEFAULT_META_ALE.keys()
    assert exp.data.keys() == DEFAULT_DATA_ALE.keys()


@pytest.mark.parametrize('extrapolate_constant', (True, False))
@pytest.mark.parametrize('extrapolate_constant_perc', (10., 50.))
@pytest.mark.parametrize('extrapolate_constant_min', (0.1, 1.0))
@pytest.mark.parametrize('constant_value', (5.,))
@pytest.mark.parametrize('feature', (1,))
def test_constant_feature(extrapolate_constant, extrapolate_constant_perc, extrapolate_constant_min,
                          constant_value, feature):
    X = np.random.normal(size=(100, 2))
    X[:, feature] = constant_value
    predict = lambda x: x.sum(axis=1)  # dummy predictor # noqa

    q, ale, ale0 = ale_num(predict, X, feature, extrapolate_constant=extrapolate_constant,
                           extrapolate_constant_perc=extrapolate_constant_perc,
                           extrapolate_constant_min=extrapolate_constant_min)
    if extrapolate_constant:
        delta = max(constant_value * extrapolate_constant_perc / 100, extrapolate_constant_min)
        assert_allclose(q, np.array([constant_value - delta, constant_value + delta]))
    else:
        assert_allclose(q, np.array([constant_value]))
        assert_allclose(ale, np.array([[0.]]))
        assert_allclose(ale0, np.array([0.]))
