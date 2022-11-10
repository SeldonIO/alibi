import re

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from alibi.explainers import (PartialDependence, PartialDependenceVariance,
                              TreePartialDependence)
from alibi.api.defaults import DEFAULT_META_PDVARIANCE, DEFAULT_DATA_PDVARIANCE


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

    for pd in pd_exp.data['pd_values']:
        cond_imp1 = np.std(pd[0], ddof=1, axis=-1)
        inter1 = np.std(cond_imp1, ddof=1, axis=-1)

        cond_imp2 = np.std(pd[0].T, ddof=1, axis=-1)
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

    for pd in pd_exp.data['pd_values']:
        cond_imp1 = (np.max(pd[0], axis=-1) - np.min(pd[0], axis=-1)) / 4
        inter1 = (np.max(cond_imp1, axis=-1) - np.min(cond_imp1, axis=-1)) / 4

        cond_imp2 = (np.max(pd[0].T, axis=-1) - np.min(pd[0].T, axis=-1)) / 4
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

    for fts, pd in zip(features, pd_exp.data['pd_values']):
        first_ft, second_ft = fts

        if second_ft in category_map:
            cond_imp1 = (np.max(pd[0], axis=-1) - np.min(pd[0], axis=-1)) / 4
            imp1 = np.std(cond_imp1, ddof=1, axis=-1)

            cond_imp2 = np.std(pd[0].T, ddof=1, axis=-1)
            imp2 = (np.max(cond_imp2, axis=-1) - np.min(cond_imp2, axis=-1)) / 4

            pd_ft_inter.append(np.mean([imp1, imp2]))
        else:
            cond_imp1 = np.std(pd[0], ddof=1, axis=-1)
            imp1 = (np.max(cond_imp1, axis=-1) - np.min(cond_imp1, axis=-1)) / 4

            cond_imp2 = (np.max(pd[0].T, axis=-1) - np.min(pd[0].T, axis=-1)) / 4
            imp2 = np.std(cond_imp2, ddof=1, axis=-1)

            pd_ft_inter.append(np.mean([imp1, imp2]))

    pd_ft_inter = np.array(pd_ft_inter)
    np.testing.assert_allclose(pdv_ft_inter, pd_ft_inter)


@pytest.fixture(scope='module')
def explanation_importance():
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
        np.array([[3.543, 3.543], [-52.780, -52.780], [-30.537, -30.537]]),
        np.array([[3.543, 3.543], [-52.780, -52.780], [-30.537, -30.537]])
    ]
    data['feature_values'] = [
        np.array([-2.319, 0.557]),
        np.array([0.080, 1.449]),
        np.array([1., 4.]),
        np.array([3., 4.])
    ]
    data['feature_names'] = ['f_0', 'f_1', 'f_2', 'f_3']
    data['feature_importance'] = np.array([
        [143.809, 83.283, 0., 0.],
        [181.973, 32.615, 0., 0.],
        [72.698, 1.212, 0., 0.]
    ])
    return Explanation(meta=meta, data=data)


@pytest.fixture(scope='module')
def explanation_interaction():
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
            [[-76.017, -76.017], [-29.543, -29.543]],
            [[-31.394, -31.394], [-29.679, -29.679]]
        ]),
        np.array([
            [[3.543, 3.543], [3.543, 3.543]],
            [[-52.780, -52.780], [-52.780, -52.780]],
            [[-30.537, -30.537], [-30.537, -30.537]]
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
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    data['conditional_importance'] = [
        [np.array([0., 0., 0.]), np.array([0., 0., 0.])],
        [np.array([0, 0, 0]), np.array([0, 0, 0])],
        [np.array([0, 0, 0]), np.array([0, 0, 0])]
    ]
    data['conditional_importance_values'] = [
        [
            np.array([[83.283, 83.283], [32.861, 32.861], [1.212, 1.212]]),
            np.array([[143.809, 143.809], [181.797, 181.797], [72.698, 72.698]])
        ],
        [
            np.array([[0., 0.], [0., 0.], [0., 0.]]),
            np.array([[83.283, 83.283], [32.861, 32.861], [1.212, 1.212]])
        ],
        [
            np.array([[0., 0.], [0., 0.], [0., 0.]]),
            np.array([[0., 0.], [0., 0.], [0., 0.0]])
        ]
    ]
    return Explanation(meta=meta, data=data)
