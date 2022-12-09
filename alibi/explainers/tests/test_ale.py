import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest_lazyfixture import lazy_fixture

from alibi.api.defaults import DEFAULT_DATA_ALE, DEFAULT_META_ALE
from alibi.api.interfaces import Explanation
from alibi.explainers.ale import (_plot_one_ale_num, adaptive_grid, ale_num,
                                  get_quantiles, minimum_satisfied, plot_ale)


@pytest.mark.parametrize('min_bin_points', [1, 4, 10])
@pytest.mark.parametrize('dataset', [lazy_fixture('diabetes_data')])
@pytest.mark.parametrize('lr_regressor',
                         [lazy_fixture('diabetes_data')],
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
def test_constant_feature(extrapolate_constant, extrapolate_constant_perc, extrapolate_constant_min,
                          constant_value, feature):
    X = np.random.normal(size=(100, 2))
    X[:, feature] = constant_value
    predict = lambda x: x.sum(axis=1)  # dummy predictor # noqa

    q, ale, ale0 = ale_num(predictor=predict,
                           X=X,
                           feature=feature,
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


@pytest.mark.parametrize('num_bins', [1, 3, 5, 7, 15])
@pytest.mark.parametrize('perc_bins', [0.1, 0.2, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize('size_data', [1, 5, 10, 50, 100])
@pytest.mark.parametrize('outside_grid', [False, True])
def test_grid_points_stress(num_bins, perc_bins, size_data, outside_grid):
    np.random.seed(0)
    eps = 1

    # define the grid between [-10, 10] having `num_bins` bins
    grid = np.unique(np.random.uniform(-10, 10, size=num_bins + 1))

    # select specific bins to sample the data from grid defined above.
    # the number of bins is controlled by the percentage of bins given by `perc_bins`
    nbins = int(np.ceil(num_bins * perc_bins))
    bins = np.sort(np.random.choice(num_bins, size=nbins, replace=False))

    # generate data
    X = []
    selected_bins = []

    for i in range(size_data):
        # select a bin at random and mark it as selected
        bin = np.random.choice(bins, size=1)
        selected_bins.append(bin.item())

        # define offset to ensure that the value is sampled within the bin
        # (i.e. avoid edge cases where  the data might land on the grid point)
        # the ALE implementation should work even in that case, only the process of constructing
        # the expected values might require additional logic
        offset = 0.1 * (grid[bin + 1] - grid[bin])
        X.append(np.random.uniform(low=grid[bin] + offset, high=grid[bin + 1] - offset).item())

    # add values outside the grid to test that the grid is extended
    if outside_grid:
        X = X + [grid[0] - eps, grid[-1] + eps]

    # construct dataset, define dummy predictor, and get grid values used by ale
    X = np.array(X).reshape(-1, 1)
    predict = lambda x: x.sum(axis=1)  # noqa
    q, _, _ = ale_num(predictor=predict, X=X, feature=0, feature_grid_points=grid)

    # construct expected grid by merging selected bins
    if outside_grid:
        # add first and last bin corresponding to min and max value.
        # This requires incrementing all the previous values by 1
        selected_bins = np.array(selected_bins + [-1, num_bins]) + 1

        # update grid point to include the min and max
        grid = np.insert(grid, 0, grid[0] - eps)
        grid = np.insert(grid, len(grid), grid[-1] + eps)

    selected_bins = np.unique(selected_bins)
    expected_q = np.array([grid[selected_bins[0]]] + [grid[b + 1] for b in selected_bins])
    np.testing.assert_allclose(q, expected_q)


@pytest.fixture(scope='module')
def explanation():
    meta = deepcopy(DEFAULT_META_ALE)
    data = deepcopy(DEFAULT_DATA_ALE)

    params = {
        'check_feature_resolution': True,
        'low_resolution_threshold': 10,
        'extrapolate_constant': True,
        'extrapolate_constant_perc': 10.0,
        'extrapolate_constant_min': 0.1,
        'min_bin_points': 4
    }

    meta.update(name='ALE', params=params)

    data['ale_values'] = [
        np.array([[0.19286464, -0.19286464],
                  [0.19154015, -0.19154015],
                  [0.10176856, -0.10176856],
                  [-0.27236635, 0.27236635],
                  [-0.61915417, 0.61915417]]),

        np.array([[0.02562475, -0.02562475],
                  [0.01918759, -0.01918759],
                  [0.01825016, -0.01825016],
                  [-0.0422308, 0.0422308],
                  [-0.060851, 0.060851]])
    ]
    data['constant_value'] = 0.5
    data['ale0'] = [
        np.array([-0.19286464, 0.19286464]),
        np.array([-0.02562475, 0.02562475])
    ]
    data['feature_values'] = [
        np.array([-1.1492519, -1.13838482, -0.4544552, 1.00972385, 2.66536596]),
        np.array([-1.16729904, -0.96888455, -0.94992442, 0.72328773, 0.98771278])
    ]
    data['feature_names'] = np.array(['f_0', 'f_1'], dtype='<U3')
    data['target_names'] = np.array(['c_0', 'c_1'], dtype='<U3')
    data['feature_deciles'] = [
        np.array([-1.1492519, -1.14490507, -1.14055824, -1.00159889, -0.72802705, -0.4544552, 0.13121642, 0.71688804,
                  1.34085227, 2.00310912, 2.66536596]),
        np.array([-1.16729904, -1.08793324, -1.00856744, -0.96509252, -0.95750847, -0.94992442, -0.28063956, 0.3886453,
                  0.77617274, 0.88194276, 0.98771278])
    ]
    return Explanation(meta=meta, data=data)


@pytest.mark.parametrize('constant', [False, True])
def test__plot_one_ale_num(explanation, constant):
    """ Test the `_plot_one_ale_num` function. """
    feature = 0
    targets = [0, 1]

    fig, ax = plt.subplots()
    ax = _plot_one_ale_num(exp=explanation,
                           feature=feature,
                           targets=targets,
                           constant=constant,
                           ax=ax,
                           legend=True,
                           line_kw={'label': None})

    x1, y1 = ax.lines[1].get_xydata().T
    x2, y2 = ax.lines[2].get_xydata().T

    assert np.allclose(x1, explanation.data['feature_values'][feature])
    assert np.allclose(x2, explanation.data['feature_values'][feature])

    expected_ale_values = explanation.data['ale_values'][feature] + constant * explanation.data['constant_value']
    assert np.allclose(y1, expected_ale_values[:, targets[0]])
    assert np.allclose(y2, expected_ale_values[:, targets[1]])

    assert ax.get_legend().texts[0].get_text() == f'c_{targets[0]}'
    assert ax.get_legend().texts[1].get_text() == f'c_{targets[1]}'
    assert ax.get_xlabel() == f'f_{feature}'

    # extract deciles form the plot
    segments = ax.collections[0].get_segments()
    deciles = [segment[0][0] for segment in segments]
    assert np.allclose(deciles, explanation.data['feature_deciles'][feature][1:])


@pytest.mark.parametrize('feats', [['f5']])
def test_plot_ale_features_error(feats, explanation):
    """ Test if an error is raised when the name of the feature does not exist. """
    with pytest.raises(ValueError) as err:
        plot_ale(exp=explanation, features=feats)
    assert f"Feature name {feats[0]} does not exist." == str(err.value)


@pytest.mark.parametrize('feats', [[0], [0, 1], ['f_0'], ['f_0', 'f_1'], 'all'])
def test_plot_ale_features(feats, explanation, mocker):
    """ Test if `plot_ale` returns the expected number of plots given by the number of features. """
    m = mocker.patch('alibi.explainers.ale._plot_one_ale_num')
    axes = plot_ale(exp=explanation, features=feats).ravel()

    if feats == 'all':
        expected_features = list(range(len(explanation.data['feature_names'])))
    else:
        expected_features = [int(re.findall(r'\d+', f)[0]) if isinstance(f, str) else f for f in feats]

    call_features = [kwargs['feature'] for _, kwargs in m.call_args_list]
    assert np.allclose(call_features, expected_features)
    assert len(axes) == len(expected_features)


@pytest.mark.parametrize('targets', [['c_5']])
def test_plot_ale_targets_error(targets, explanation):
    """ Test if an error is raised when the name of the target does not exist. """
    with pytest.raises(ValueError) as err:
        plot_ale(exp=explanation, targets=targets)
    assert f"Target name {targets[0]} does not exist." == str(err.value)


@pytest.mark.parametrize('targets', [[0], [0, 1], ['c_0'], ['c_0', 'c_1'], 'all'])
def test_plot_ale_targets(targets, explanation, mocker):
    """ Test if `plot_ale` plots all the given targets. """
    m = mocker.patch('alibi.explainers.ale._plot_one_ale_num')
    plot_ale(exp=explanation, targets=targets).ravel()

    _, kwargs = m.call_args
    call_targets = kwargs['targets']

    if targets == 'all':
        expected_targets = list(range(len(explanation.data['target_names'])))
    else:
        expected_targets = [int(re.findall(r'\d+', t)[0]) if isinstance(t, str) else t for t in targets]

    assert np.allclose(call_targets, expected_targets)


@pytest.mark.parametrize('n_cols', [1, 2, 3])
def test_plot_ale_n_cols(n_cols, explanation, mocker):
    """ Test if the number of plot columns matches the expectation. """
    mocker.patch('alibi.explainers.ale._plot_one_ale_num')
    axes = plot_ale(exp=explanation, features=[0, 0, 0], n_cols=n_cols)
    assert axes.shape[-1] == n_cols


def test_plot_ale_sharey_all(explanation):
    """ Test if all axes have the same y limits when ``sharey='all'``. """
    axes = plot_ale(exp=explanation, features=[0, 1], n_cols=1, sharey='all').ravel()
    assert len(set([ax.get_ylim() for ax in axes])) == 1


@pytest.mark.parametrize('n_cols', [1, 2])
def test_plot_ale_sharey_row(n_cols, explanation):
    """ Test if all axes on the same rows have the same y limits and axes on different rows have different y limits
    when ``sharey=row``. """
    axes = plot_ale(exp=explanation, features=[0, 1], n_cols=n_cols, sharey='row')

    if n_cols == 1:
        # different rows should have different y-limits
        assert axes[0, 0].get_ylim() != axes[1, 0].get_ylim()
    else:
        # same row should have the same y-limits
        assert axes[0, 0].get_ylim() == axes[0, 1].get_ylim()


@pytest.mark.parametrize('n_cols', [1, 2])
def test_plot_ale_sharey_none(n_cols, explanation):
    """ Test if all axes have different y limits when ``sharey=None``. """
    axes = plot_ale(exp=explanation, features=[0, 1], n_cols=n_cols, sharey=None).ravel()
    assert axes[0].get_ylim() != axes[1].get_ylim()


def test_plot_ale_axes_error(explanation):
    """ Test if an error is raised when the number of provided axes is less that the number of features. """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2)
    feats = [0, 0, 0]

    with pytest.raises(ValueError) as err:
        plot_ale(exp=explanation, features=feats, ax=axes)
    assert f"Expected ax to have {len(features)} axes, got {axes.size}" == str(err.value)


@pytest.mark.parametrize('label', [None, ['target_1', 'target_2']])
def test_plot_ale_legend(label, explanation):
    """ Test if the legend is displayed only for the first ax object with the expected text. """
    axes = plot_ale(exp=explanation, line_kw={'label': label}).ravel()
    assert axes[0].get_legend() is not None
    assert all([ax.get_legend() is None for ax in axes[1:]])

    texts = [text.get_text() for text in axes[0].get_legend().get_texts()]
    if label is None:
        assert texts == explanation.data['target_names'].tolist()
    else:
        assert texts == label
