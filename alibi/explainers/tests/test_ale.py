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
num_gridpoints = [1, 2, 4, 8, 32, 64, 128]


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
@pytest.mark.parametrize('custom_grid, num_grid_points', [
    (False, None), (True, 1), (True, 2), (True, 4), (True, 8), (True, 32), (True, 128)
], ids='custom_grid={}'.format)
def test_explain(mock_ale_explainer, features, input_dim, batch_size, custom_grid, num_grid_points):
    out_dim = mock_ale_explainer.predictor.out_dim
    X = np.random.rand(batch_size, input_dim)

    if features:
        n_features = len(features)
        if not custom_grid:
            grid_points = None
        else:
            grid_points = {f: np.random.rand(num_grid_points) for f in features}
    else:
        n_features = input_dim
        if not custom_grid:
            grid_points = None
        else:
            grid_points = {f: np.random.rand(num_grid_points) for f in range(n_features)}

    exp = mock_ale_explainer.explain(X, features=features, grid_points=grid_points)

    # check that the length of all relevant attributes is the same as the number of features explained
    assert all(len(attr) == n_features for attr in (exp.ale_values, exp.feature_values,
                                                    exp.feature_names, exp.feature_deciles,
                                                    exp.ale0))

    assert len(exp.target_names) == out_dim
    for alev, featv in zip(exp.ale_values, exp.feature_values):
        assert alev.shape == (featv.shape[0], out_dim)

    if custom_grid:
        for i, f in enumerate(grid_points.keys()):
            # need to remove the first and last element just in case the feature values are extended
            # with the min & max feature value. Check if subset since some grid-points might have
            # been removed because of merging empty intervals.
            assert np.all(np.isin(exp.feature_values[i][1:-1], grid_points[f]))

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
@pytest.mark.parametrize('custom_grid', (True, False))
def test_constant_feature(extrapolate_constant, extrapolate_constant_perc, extrapolate_constant_min,
                          constant_value, feature, custom_grid):
    X = np.random.normal(size=(100, 2))
    X[:, feature] = constant_value
    predict = lambda x: x.sum(axis=1)  # dummy predictor # noqa
    feature_grid_points = np.random.normal((1, )) if custom_grid else None

    q, ale, ale0 = ale_num(predictor=predict,
                           X=X,
                           feature=feature,
                           feature_grid_points=feature_grid_points,
                           extrapolate_constant=extrapolate_constant,
                           extrapolate_constant_perc=extrapolate_constant_perc,
                           extrapolate_constant_min=extrapolate_constant_min)
    if extrapolate_constant:
        delta = max(constant_value * extrapolate_constant_perc / 100, extrapolate_constant_min)
        assert_allclose(q, np.array([constant_value - delta, constant_value + delta]))
    else:
        assert_allclose(q, np.array([constant_value]))
        assert_allclose(ale, np.array([[0.]]))
        assert_allclose(ale0, np.array([0.]))
