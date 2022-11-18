import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from alibi.api.defaults import DEFAULT_DATA_PDVARIANCE, DEFAULT_META_PDVARIANCE
from alibi.api.interfaces import Explanation
from alibi.explainers import (PartialDependence, PartialDependenceVariance,
                              TreePartialDependence)
from alibi.explainers.pd_variance import (_plot_feature_importance,
                                          _plot_feature_interaction,
                                          _plot_hbar, plot_pd_variance)


@pytest.mark.parametrize('predictor', [LinearRegression(), GradientBoostingRegressor()])
def test_blackbox_pd(predictor):
    """ Tests whether the `pd_explainer` initialized within the
    `PartialDependenceVariance` explainer is  of type `PartialDependence`. """
    explainer = PartialDependenceVariance(predictor=predictor.predict)
    assert isinstance(explainer.pd_explainer, PartialDependence)


@pytest.mark.parametrize('predictor', [GradientBoostingRegressor()])
def test_tree_pd(predictor, iris_data):
    """ Tests whether the `pd_explainer` initialized within the
    `PartialDependenceVariance` explainer is of type `TreePartialDependence`. """
    predictor = predictor.fit(iris_data['X_train'], iris_data['y_train'])
    explainer = PartialDependenceVariance(predictor=predictor)
    assert isinstance(explainer.pd_explainer, TreePartialDependence)


@pytest.mark.parametrize('method', ['unknown'])
def test_unknown_method(method):
    """ Test whether an unknown method raises an error. """
    explainer = PartialDependenceVariance(predictor=lambda x: np.random.empty(len(x)))
    with pytest.raises(ValueError) as err:
        explainer.explain(X=np.empty((10, 3)), method=method)
    assert re.search('unknown method', err.value.args[0].lower())


@pytest.mark.parametrize('method', ['importance', 'interaction'])
@pytest.mark.parametrize('features', [[0, 1, (0, 1)]])
def test_invalid_features(method, features):
    """ Tests whether an invalid list of features raises an error. """
    explainer = PartialDependenceVariance(predictor=lambda x: np.empty((len(x), )))
    with pytest.raises(ValueError) as err:
        explainer.explain(X=np.empty((10, 3)), method=method, features=features)
    msg = 'all features must be integers' if method == 'importance' else 'all features must be tuples of length 2'
    assert re.search(msg, err.value.args[0].lower())


@pytest.mark.parametrize('num_targets', [1, 2, 5])
@pytest.mark.parametrize('num_features', [2, 3, 5, 10])
@pytest.mark.parametrize('method', ['importance', 'interaction'])
def test_explanation_shapes(num_targets, num_features, method):
    """ Tests whether the explanation shapes match the expectation. """
    explainer = PartialDependenceVariance(predictor=lambda x: np.empty((len(x), num_targets)))
    exp = explainer.explain(X=np.empty((10, num_features)), method=method)
    expected_num_features = num_features if method == 'importance' else num_features * (num_features - 1) // 2
    assert len(exp.data["feature_deciles"]) == expected_num_features
    assert len(exp.data["feature_values"]) == expected_num_features
    assert len(exp.data["feature_names"]) == expected_num_features
    assert len(exp.data["pd_values"]) == expected_num_features
    assert np.all(pd_values.shape[0] == num_targets for pd_values in exp.data['pd_values'])

    if method == 'importance':
        assert exp.data["feature_importance"].shape == (num_targets, expected_num_features)
    else:
        assert exp.data["feature_interaction"].shape == (num_targets, expected_num_features)
        assert len(exp.data["conditional_importance"]) == expected_num_features
        assert len(exp.data["conditional_importance"]) == expected_num_features


def test_zero_importance():
    """ Test whether the explainer attributes a zero importance to each feature for a constant predictor. """
    explainer = PartialDependenceVariance(predictor=lambda x: np.full((len(x),), fill_value=1))
    exp = explainer.explain(X=np.random.rand(100, 10), method='importance')
    assert np.allclose(exp.data['feature_importance'], 0)


@pytest.mark.parametrize('predictor', [
    lambda x: x[:, 0] + x[:, 1],
    lambda x: x[:, 0] * x[:, 1] > 0
])
def test_zero_interaction(predictor):
    """ Test whether the explainer attributes zero feature interaction. """
    X = np.random.uniform(low=-1, high=1, size=(100, 2))
    points = np.linspace(-0.9, 0.9, 10)
    grid_points = {i: points for i in range(X.shape[1])}

    explainer = PartialDependenceVariance(predictor=predictor)
    exp = explainer.explain(X=X, method='interaction', grid_points=grid_points)
    assert np.allclose(exp.data['feature_interaction'], 0)


@pytest.mark.parametrize('num_features', [2, 3, 5, 10])
def test_linear_regression(num_features):
    """ Test whether the order of the feature based on the feature importance is the one expected for a linear
    regression model. """
    X = np.random.uniform(low=-1, high=1, size=(100, num_features))
    w = np.random.randn(num_features)

    points = np.linspace(-1, 1, 21)
    grid_points = {i: points for i in range(num_features)}

    explainer = PartialDependenceVariance(predictor=lambda x: x @ w)
    exp = explainer.explain(X=X, method='importance', grid_points=grid_points)

    expected_order = np.argsort(np.abs(w))
    order = np.argsort(exp.data['feature_importance'][0])
    assert np.all(expected_order == order)


@pytest.fixture(scope='module')
def pd_pdv_explainers(request, adult_data):
    """ Initialize `PartialDependence` and `PartialDependenceVariance` explainers. """
    clf, preprocessor = request.param
    feature_names = adult_data['metadata']['feature_names']
    categorical_names = adult_data['metadata']['category_map']

    # define predictor function with a single output to be used
    def predictor(x):
        return clf.predict_proba(preprocessor.transform(x))[:, 1:]

    # compute manually the feature importance using the `PartialDependence` explainer
    pd_explainer = PartialDependence(predictor=predictor,
                                     feature_names=feature_names,
                                     categorical_names=categorical_names)

    # compute feature importance using the `PartialDependenceVariance` explainer.
    pdv_explainer = PartialDependenceVariance(predictor=predictor,
                                              feature_names=feature_names,
                                              categorical_names=categorical_names)
    return pd_explainer, pdv_explainer


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('pd_pdv_explainers', [lazy_fixture('rf_classifier')], indirect=True)
@pytest.mark.parametrize('features', [[0, 8, 9]])
def test_importance_num(rf_classifier, pd_pdv_explainers, features, adult_data):
    """ Test the computation of the feature importance for a numerical feature. """
    X_train = adult_data['X_train'][:100]
    pd_explainer, pdv_explainer = pd_pdv_explainers

    # compute feature importance using the `PartialDependenceVariance` explainer.
    pdv_exp = pdv_explainer.explain(X=X_train, features=features)
    pdv_ft_imp = pdv_exp.data['feature_importance'][0]

    # compute manually the feature importance using the `PartialDependence` explainer.
    pd_exp = pd_explainer.explain(X=X_train, features=features)
    pd_ft_imp = np.array([np.std(pd[0], ddof=1, axis=-1) for pd in pd_exp.data['pd_values']])
    np.testing.assert_allclose(pdv_ft_imp, pd_ft_imp)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('pd_pdv_explainers', [lazy_fixture('rf_classifier')], indirect=True)
@pytest.mark.parametrize('features', [[1, 2, 3, 4]])
def test_importance_cat(rf_classifier, pd_pdv_explainers, features, adult_data):
    """ Test the computation of the feature importance for a categorical feature. """
    X_train = adult_data['X_train'][:100]
    pd_explainer, pdv_explainer = pd_pdv_explainers

    # compute feature importance using the `PartialDependenceVariance` explainer
    pdv_exp = pdv_explainer.explain(X=X_train, features=features)
    pdv_ft_imp = pdv_exp.data['feature_importance'][0]

    # compute manually the feature importance using the `PartialDependence` explainer
    pd_exp = pd_explainer.explain(X=X_train, features=features)
    pd_ft_imp = np.array([(np.max(pd[0], axis=-1) - np.min(pd[0], axis=-1)) / 4 for pd in pd_exp.data['pd_values']])
    np.testing.assert_allclose(pdv_ft_imp, pd_ft_imp)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('pd_pdv_explainers', [lazy_fixture('rf_classifier')], indirect=True)
@pytest.mark.parametrize('features', [[(0, 8), (0, 9), (8, 9)]])
def test_interaction_num_num(rf_classifier, pd_pdv_explainers, features, adult_data):
    """ Test the computation of the feature interaction for two numerical features. """
    X_train = adult_data['X_train'][:100]
    pd_explainer, pdv_explainers = pd_pdv_explainers

    # compute feature interaction using the `PartialDependenceVariance` explainer
    pdv_exp = pdv_explainers.explain(X=X_train, method='interaction', features=features, grid_resolution=20)
    pdv_ft_inter = pdv_exp.data['feature_interaction'][0]

    # compute manually the feature interaction using the `PartialDependence` explainer
    pd_exp = pd_explainer.explain(X=X_train, features=features, grid_resolution=20)
    pd_ft_inter = []

    for pd_vals in pd_exp.data['pd_values']:
        cond_imp1 = np.std(pd_vals[0], ddof=1, axis=-1)
        inter1 = np.std(cond_imp1, ddof=1, axis=-1)

        cond_imp2 = np.std(pd_vals[0].T, ddof=1, axis=-1)
        inter2 = np.std(cond_imp2, ddof=1, axis=-1)

        pd_ft_inter.append(np.mean([inter1, inter2]))

    pd_ft_inter = np.array(pd_ft_inter)
    np.testing.assert_allclose(pdv_ft_inter, pd_ft_inter)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('pd_pdv_explainers', [lazy_fixture('rf_classifier')], indirect=True)
@pytest.mark.parametrize('features', [[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]])
def test_interaction_cat_cat(rf_classifier, pd_pdv_explainers, features, adult_data):
    """ Test the computation of the feature interaction for two categorical features. """
    X_train = adult_data['X_train'][:100]
    pd_explainer, pdv_explainer = pd_pdv_explainers

    # compute feature interaction using the `PartialDependenceVariance` explainer
    pdv_exp = pdv_explainer.explain(X=X_train, method='interaction', features=features)
    pdv_ft_inter = pdv_exp.data['feature_interaction'][0]

    # compute manually the feature interaction using the `PartialDependence` explainer
    pd_exp = pd_explainer.explain(X=X_train, features=features)
    pd_ft_inter = []

    for pd_vals in pd_exp.data['pd_values']:
        cond_imp1 = (np.max(pd_vals[0], axis=-1) - np.min(pd_vals[0], axis=-1)) / 4
        inter1 = (np.max(cond_imp1, axis=-1) - np.min(cond_imp1, axis=-1)) / 4

        cond_imp2 = (np.max(pd_vals[0].T, axis=-1) - np.min(pd_vals[0].T, axis=-1)) / 4
        inter2 = (np.max(cond_imp2, axis=-1) - np.min(cond_imp2, axis=-1)) / 4

        pd_ft_inter.append(np.mean([inter1, inter2]))

    pd_ft_inter = np.array(pd_ft_inter)
    np.testing.assert_allclose(pdv_ft_inter, pd_ft_inter)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('pd_pdv_explainers', [lazy_fixture('rf_classifier')], indirect=True)
@pytest.mark.parametrize('features', [[(0, 1), (1, 0), (8, 2), (2, 8), (3, 9), (9, 3)]])
def test_interaction_num_cat(rf_classifier, pd_pdv_explainers, features, adult_data):
    """ Test the computation of the feature interaction for a numerical and a categorical feature. """
    X_train = adult_data['X_train'][:100]
    category_map = adult_data['metadata']['category_map']
    pd_explainer, pdv_explainer = pd_pdv_explainers

    # compute feature interaction using the `PartialDependenceVariance` explainer
    pdv_exp = pdv_explainer.explain(X=X_train, method='interaction', features=features, grid_resolution=20)
    pdv_ft_inter = pdv_exp.data['feature_interaction'][0]

    # compute manually the feature interaction using the `PartialDependence` explainer
    pd_exp = pd_explainer.explain(X=X_train, features=features, grid_resolution=20)
    pd_ft_inter = []

    for fts, pd_vals in zip(features, pd_exp.data['pd_values']):
        first_ft, second_ft = fts

        if second_ft in category_map:
            cond_imp1 = (np.max(pd_vals[0], axis=-1) - np.min(pd_vals[0], axis=-1)) / 4
            imp1 = np.std(cond_imp1, ddof=1, axis=-1)

            cond_imp2 = np.std(pd_vals[0].T, ddof=1, axis=-1)
            imp2 = (np.max(cond_imp2, axis=-1) - np.min(cond_imp2, axis=-1)) / 4

            pd_ft_inter.append(np.mean([imp1, imp2]))
        else:
            cond_imp1 = np.std(pd_vals[0], ddof=1, axis=-1)
            imp1 = (np.max(cond_imp1, axis=-1) - np.min(cond_imp1, axis=-1)) / 4

            cond_imp2 = (np.max(pd_vals[0].T, axis=-1) - np.min(pd_vals[0].T, axis=-1)) / 4
            imp2 = np.std(cond_imp2, ddof=1, axis=-1)

            pd_ft_inter.append(np.mean([imp1, imp2]))

    pd_ft_inter = np.array(pd_ft_inter)
    np.testing.assert_allclose(pdv_ft_inter, pd_ft_inter)


@pytest.fixture(scope='module')
def exp_importance():
    """ Creates importance explanation object. """
    meta = deepcopy(DEFAULT_META_PDVARIANCE)
    data = deepcopy(DEFAULT_DATA_PDVARIANCE)
    meta['params'] = {
        'percentiles': (0.0, 1.0),
        'grid_resolution': 4,
        'feature_names': ['f_0', 'f_1', 'f_2', 'f_3'],
        'categorical_names': {
            2: [0, 1, 2, 3, 4],
            3: [0, 1, 2, 3, 4, 5, 6]
        },
        'target_names': ['c_0', 'c_1'],
        'method': 'importance',
    }
    data['feature_deciles'] = [
        np.array([-2.319, -2.032, -1.744, -1.456, -1.169, -0.881, -0.593, -0.305, -0.018, 0.269, 0.557]),
        np.array([0.080, 0.217, 0.354, 0.491, 0.628, 0.765, 0.902, 1.039, 1.175, 1.312, 1.449]),
        None,
        None
    ]
    data['pd_values'] = [
        np.array([[-98.144, 105.232], [-181.330, 75.769], [-81.942, 20.868]]),
        np.array([[-55.346, 62.434], [-76.017, -29.543], [-31.394, -29.679]]),
        np.array([[90.638, 3.543], [-52.780, 12.725], [-30.537, 32.840]]),
        np.array([[13.115, 54.495], [-28.445, 41.850], [-11.403, -6.029]])
    ]
    data['feature_values'] = [
        np.array([-2.319, 0.557]),
        np.array([0.080, 1.449]),
        np.array([1., 4.]),
        np.array([3., 4.])
    ]
    data['feature_names'] = ['f_0', 'f_1', 'f_2', 'f_3']
    data['feature_importance'] = np.array([
        [143.809, 83.283, 77.481, 17.177],
        [181.973, 32.615, 32.840, 11.403],
        [72.698, 1.212, 41.693, 10.782]
    ])
    return Explanation(meta=meta, data=data)


@pytest.fixture(scope='module')
def exp_interaction():
    """ Creates interaction explanation object. """
    meta = deepcopy(DEFAULT_META_PDVARIANCE)
    data = deepcopy(DEFAULT_DATA_PDVARIANCE)

    meta['params'] = {
        'percentiles': (0.0, 1.0),
        'grid_resolution': 4,
        'feature_names': ['f_0', 'f_1', 'f_2', 'f_3'],
        'categorical_names': {
            2: [0, 1, 2, 3, 4],
            3: [0, 1, 2, 3, 4, 5, 6]
        },
        'target_names': ['c_0', 'c_1'],
        'method': 'interaction'
    }
    data['feature_deciles'] = [
        [
            np.array([-2.319, -2.032, -1.744, -1.456, -1.169, -0.881, -0.593, -0.305, -0.018, 0.269, 0.557]),
            np.array([0.080, 0.217, 0.354, 0.491, 0.628, 0.765, 0.902, 1.039, 1.175, 1.312, 1.449])
        ],
        [
            np.array([0.080, 0.217, 0.354, 0.491, 0.628, 0.765, 0.902, 1.039, 1.175, 1.312, 1.449]),
            None
        ],
        [
            None,
            None
        ]
    ]
    data['pd_values'] = [
        np.array([
            [[-157.035, -39.254], [46.342, 164.122]],
            [[-204.567, -158.093], [52.533, 99.006]],
            [[-82.799, -81.085], [20.010, 21.725]]
        ]),
        np.array([
            [[-55.346, -55.346], [62.434, 62.434]],
            [[-76.017, -51.231], [-12.543, -29.543]],
            [[-31.394, -67.631], [-29.679, -4.852]]
        ]),
        np.array([
            [[3.543, 5.143], [6.123, 12.755]],
            [[-52.214, -1.523], [-27.732, -43.564]],
            [[-13.537, -35.537], [-5.123, -30.537]]
        ])
    ]
    data['feature_values'] = [
        [
            np.array([-2.319, 0.557]),
            np.array([0.080, 1.449])
        ],
        [
            np.array([0.080, 1.449]),
            np.array([1., 4.])
        ],
        [
            np.array([1., 4.]),
            np.array([0., 4.])
        ]
    ]
    data['feature_names'] = [('f_0', 'f_1'), ('f_1', 'f_2'), ('f_2', 'f_3')]
    data['feature_importance'] = None
    data['feature_interaction'] = np.array([
        [1., 3., 2.],
        [5., 1., 4.],
        [1., 3., 2.]
    ])
    data['conditional_importance'] = [
        [np.array([1., 5., 4.]), np.array([2., 3., 1.])],
        [np.array([7., 2., 1.]), np.array([3., 2., 1.])],
        [np.array([6., 3., 3.]), np.array([4., 1., 5.])]
    ]
    data['conditional_importance_values'] = [
        [
            np.array([[43.879, 24.657], [91.827, 32.861], [29.467, 1.212]]),
            np.array([[59.551, 143.809], [181.797, 29.227], [91.411, 72.698]])
        ],
        [
            np.array([[72.121, 32.357], [21.731, 13.441], [53.137, 89.321]]),
            np.array([[83.283, 70.373], [32.861, 34.186], [29.601, 30.791]])
        ],
        [
            np.array([[19.919, 10.809], [14.462, 50.703], [70.373, 35.073]]),
            np.array([[65.962, 18.605], [11.311, 14.507], [21.231, 24.657]])
        ]
    ]
    return Explanation(meta=meta, data=data)


@pytest.mark.parametrize('n_axes', [4, 5, 6, 7])
@pytest.mark.parametrize('n_cols', [2, 3, 4])
def test__plot_hbar_n_cols(n_axes, n_cols, exp_importance):
    """ Test if the number of axes columns matches the expectation. """
    n_rows = n_axes // n_cols + (n_axes % n_cols != 0)
    targets = np.random.choice(len(exp_importance.meta['params']['target_names'],), size=n_axes, replace=True)
    features = np.arange(len(exp_importance.meta['params']['feature_names'])).tolist()

    # create the horizontal bar plot
    ax = _plot_hbar(exp_values=exp_importance.data['feature_importance'],
                    exp_feature_names=exp_importance.data['feature_names'],
                    exp_target_names=exp_importance.meta['params']['target_names'],
                    features=features,
                    targets=targets,
                    n_cols=n_cols)

    assert ax.shape == (n_rows, n_cols)
    assert np.sum(~pd.isna(ax.ravel())) == n_axes


def test__plot_hbar_ax(exp_importance):
    """ Test if an error is raised when the number of provided axes is less than the number of targets
    to plot the horizontal bar for. """
    n_targets = 7
    targets = np.random.choice(len(exp_importance.meta['params']['target_names']), size=n_targets, replace=True)
    features = np.arange(len(exp_importance.meta['params']['feature_names'])).tolist()
    _, ax = plt.subplots(nrows=2, ncols=2)

    with pytest.raises(ValueError) as err:
        _plot_hbar(exp_values=exp_importance.data['feature_importance'],
                   exp_feature_names=exp_importance.data['feature_names'],
                   exp_target_names=exp_importance.meta['params']['target_names'],
                   features=features,
                   targets=targets,
                   ax=ax)

    assert 'Expected ax to have' in str(err.value)


@pytest.mark.parametrize('sort, top_k', [(False, None), (True, 1), (True, 3)])
@pytest.mark.parametrize('target', [0, 1])
@pytest.mark.parametrize('features', [[0, 2], [0, 1, 2], [0, 1, 2, 3]])
def test__plot_hbar_values(sort, top_k, target, features, exp_importance):
    """ Test if the plotted values, labels and titles are correct on the bar plot. """
    ax = _plot_hbar(exp_values=exp_importance.data['feature_importance'],
                    exp_feature_names=exp_importance.data['feature_names'],
                    exp_target_names=exp_importance.meta['params']['target_names'],
                    targets=[target],
                    features=features,
                    sort=sort,
                    top_k=top_k).ravel()

    datavalues = ax[0].containers[0].datavalues
    expected_datavalues = np.array([exp_importance.data['feature_importance'][target][ft] for ft in features])

    feature_names = [txt.get_text() for txt in ax[0].get_yticklabels()]
    expected_feature_names = [exp_importance.data['feature_names'][ft] for ft in features]

    if sort:
        sorted_idx = np.argsort(expected_datavalues)[::-1][:top_k]
        expected_datavalues = expected_datavalues[sorted_idx]
        expected_feature_names = [expected_feature_names[i] for i in sorted_idx]

    assert np.allclose(datavalues, expected_datavalues)
    assert feature_names == expected_feature_names
    assert ax[0].get_title() == exp_importance.meta['params']['target_names'][target]


def extract_number(x: str):
    """ Helper function to extract number from a string. """
    return float(re.findall('[0-9]+.[0-9]+', x)[0])


@pytest.mark.parametrize('sort, top_k', [(False, None), (True, 1), (True, 3)])
@pytest.mark.parametrize('features', [[0, 2], [0, 1, 2], [0, 1, 2, 3]])
@pytest.mark.parametrize('targets', [[0]])
def test__plot_feature_importance_detailed(sort, top_k, features, targets, exp_importance):
    """ Tests if the `_plot_feature_importance` returns the correct plots when ``summarise='False'``. """
    axes = _plot_feature_importance(exp=exp_importance,
                                    features=features,
                                    targets=targets,
                                    summarise=False,
                                    sort=sort,
                                    top_k=top_k).ravel()

    expected_importance = np.array([exp_importance.data['feature_importance'][targets[0]][ft] for ft in features])
    if sort:
        sorted_idx = np.argsort(expected_importance)[::-1]
        expected_importance = expected_importance[sorted_idx][:top_k]

    importance = np.array([extract_number(ax.get_title()) for ax in axes if ax is not None])
    assert np.allclose(importance, expected_importance, atol=1e-2)


def test__plot_feature_importance_summarise(exp_importance, mocker):
    """ Test if the `_plot_feature_importance` calls `_plot_hbar` once. """
    features, targets = [0, 1, 2], [0]
    m = mocker.patch('alibi.explainers.pd_variance._plot_hbar')
    _plot_feature_importance(exp=exp_importance,
                             features=features,
                             targets=targets,
                             summarise=True)
    m.assert_called_once()


@pytest.mark.parametrize('features', [[0], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize('targets', [[0]])
@pytest.mark.parametrize('sort, top_k', [(False, None), (True, 1), (True, 3)])
def test__plot_feature_interaction_detailed(features, targets, sort, top_k, exp_interaction):
    axes = _plot_feature_interaction(exp=exp_interaction,
                                     features=features,
                                     targets=targets,
                                     summarise=False,
                                     sort=sort,
                                     top_k=top_k).ravel()

    expected_interaction = np.array([
        exp_interaction.data['feature_interaction'][targets[0]][ft] for ft in features
    ])
    expected_cond_import0 = np.array([
        exp_interaction.data['conditional_importance'][ft][0][targets[0]] for ft in features
    ])
    expected_cond_import1 = np.array([
        exp_interaction.data['conditional_importance'][ft][1][targets[0]] for ft in features
    ])

    if sort:
        sorted_idx = np.argsort(expected_interaction)[::-1]
        expected_interaction = expected_interaction[sorted_idx][:top_k]
        expected_cond_import0 = expected_cond_import0[sorted_idx][:top_k]
        expected_cond_import1 = expected_cond_import1[sorted_idx][:top_k]

    interaction = [extract_number(axes[i].get_title()) for i in range(0, len(axes), 3)if axes[i] is not None]
    cond_import0 = [extract_number(axes[i].get_title()) for i in range(1, len(axes), 3) if axes[i] is not None]
    cond_import1 = [extract_number(axes[i].get_title()) for i in range(2, len(axes), 3) if axes[i] is not None]

    assert np.allclose(interaction, expected_interaction)
    assert np.allclose(cond_import0, expected_cond_import0)
    assert np.allclose(cond_import1, expected_cond_import1)


def test__plot_feature_interaction_summarise(exp_interaction, mocker):
    """ Test if the `_plot_feature_interaction` calls `_plot_hbar` once. """
    features, targets = [0, 1], [0]
    m = mocker.patch('alibi.explainers.pd_variance._plot_hbar')
    _plot_feature_interaction(exp=exp_interaction,
                              features=features,
                              targets=targets,
                              summarise=True)
    m.assert_called_once()


def test_plot_pd_variance_top_k_error():
    """ Test if an error is raised when ``sorted='True'`` and ``top_k < 0``. """
    with pytest.raises(ValueError) as err:
        plot_pd_variance(exp=None, sort=True, top_k=-1)
    assert "``top_k`` must be greater than 0." == str(err.value)


def test_plot_pd_variance_targets_all(exp_importance, mocker):
    """ Test if all the targets are considered when ``targets='all'``. """
    m = mocker.patch('alibi.explainers.pd_variance._plot_feature_importance', return_value=None)
    plot_pd_variance(exp=exp_importance, targets='all')
    args, kwargs = m.call_args
    assert kwargs['targets'] == exp_importance.meta['params']['target_names']


def test_plot_pd_variance_targets_type(exp_importance):
    """ Test if an error is raised if `targets` is not of type `list`. """
    with pytest.raises(ValueError) as err:
        plot_pd_variance(exp=exp_importance, targets=0)
    assert '`targets` must be a list.' == str(err.value)


def test_plot_pd_variance_warning(exp_importance, mocker, caplog):
    """ Tests if a warning is raise when the ``summarise=False`` and the length of `targets` is > 1. """
    mocker.patch('alibi.explainers.pd_variance._plot_feature_importance', return_value=None)
    plot_pd_variance(exp=exp_importance, targets=[0, 1], summarise=False)
    assert "`targets` should be a list containing a single element" in caplog.records[0].message


def test_plot_pd_variance_features_all(exp_importance, mocker):
    m = mocker.patch('alibi.explainers.pd_variance._plot_feature_importance', return_value=None)
    plot_pd_variance(exp=exp_importance, features='all')
    args, kwargs = m.call_args
    assert kwargs['features'] == np.arange(len(exp_importance.data['feature_names'])).tolist()


def test_plot_pd_variance_targets_unknown(exp_importance):
    """ Test if an error is raised when the ``targets`` contains an unknown values. """
    with pytest.raises(ValueError) as err:
        plot_pd_variance(exp=exp_importance, targets=['unknown'])
    assert "Unknown `target` name." in str(err.value)


def test_plot_pd_variance_targets_oor(exp_importance):
    """ Test if an error is raised when the ``targets`` contains out of range values. """
    with pytest.raises(IndexError) as err:
        plot_pd_variance(exp=exp_importance, targets=[5])
    assert "Target index out of range." in str(err.value)


def test_plot_pd_variance_importance(exp_importance, mocker):
    """ Test if `_plot_feature_importance` is called within the `plot_pd_variance` if an importance explanation
    is provided. """
    m = mocker.patch('alibi.explainers.pd_variance._plot_feature_importance', return_value=None)
    plot_pd_variance(exp=exp_importance)
    m.assert_called_once()


def test_plot_pd_variance_interaction(exp_interaction, mocker):
    """ Test if `_plot_feature_interaction` is called within the `plot_pd_variance` if an interaction explanation
    is provided. """
    m = mocker.patch('alibi.explainers.pd_variance._plot_feature_interaction', return_value=None)
    plot_pd_variance(exp=exp_interaction)
    m.assert_called_once()
