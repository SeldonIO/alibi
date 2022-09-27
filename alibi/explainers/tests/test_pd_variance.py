import re

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from alibi.explainers import (PartialDependence, PartialDependenceVariance,
                              TreePartialDependence)


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
