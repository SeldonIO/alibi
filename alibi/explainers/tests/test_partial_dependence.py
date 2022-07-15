import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from alibi.explainers import PartialDependence
from alibi.explainers.partial_dependence import ResponseMethod, Kind, Method

from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputClassifier


@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unfitted_estimator(predictor, iris_data):
    """ Checks if raises error for unfitted model. """
    X_train = iris_data['X_train']
    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(NotFittedError):
        explainer.explain(X=X_train, features_list=[0, 1, 2])


@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_classes, random_state',
    [(10, 100, 30, 3, 1)]
)
def test_multioutput_estimator(predictor, n_samples, n_features, n_informative, n_classes, random_state):
    """ Check if raises error for multi-output model"""
    X, y1 = make_classification(n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                n_classes=n_classes,
                                random_state=random_state)

    y2 = shuffle(y1, random_state=1)
    y3 = shuffle(y1, random_state=2)
    Y = np.vstack((y1, y2, y3)).T

    multi_target_predictor = MultiOutputClassifier(predictor)
    multi_target_predictor.fit(X, Y)

    explainer = PartialDependence(predictor=multi_target_predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X, features_list=[1, 2, 3])


@pytest.mark.parametrize('response_method', ['unknown'])
@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unknown_response_method(predictor, response_method, iris_data):
    """ Checks if raises error for unknown response_method. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X_train, features_list=[0, 1, 2], response_method=response_method)


@pytest.mark.parametrize('response_method', [ResponseMethod.DECISION_FUNCTION, ResponseMethod.PREDICT_PROBA])
@pytest.mark.parametrize('predictor', [RandomForestRegressor()])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_targets, random_state',
    [(10, 100, 30, 1, 1)]
)
def test_estimator_response_method(predictor, n_samples, n_features, n_informative, n_targets,
                                   random_state, response_method):
    """ Checks if raises error for a regressor with a response_method != 'auto'. """
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_targets=n_targets,
                           random_state=random_state)
    predictor.fit(X, y)

    explainer = PartialDependence(predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X, features_list=[0, 1, 2], response_method=response_method)


@pytest.mark.parametrize('method', ['unknown'])
@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unknown_method(predictor, iris_data, method):
    """ Checks if raises error for unknown method. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X_train, features_list=[0, 1, 2], method=method)


@pytest.mark.parametrize('kind', [Kind.INDIVIDUAL, Kind.BOTH])
@pytest.mark.parametrize('method', [Method.RECURSION])
@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_kind_method(predictor, iris_data, kind, method):
    """ Checks if raises error when method='recursion' and kind !='average'. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X_train, features_list=[0, 1, 2], kind=kind, method=method)


@pytest.mark.parametrize('predictor', [
    GradientBoostingRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_targets, random_state',
    [(10, 100, 30, 1, 1)]
)
def test_method_auto_recursion(predictor, n_samples, n_features, n_informative, n_targets, random_state):
    """ Checks if the method='auto' falls to method='recursion' for some specific class of regressors. """
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_targets=n_targets,
                           random_state=random_state)
    predictor.fit(X, y)

    explainer = PartialDependence(predictor=predictor)
    explanation = explainer.explain(X=X, features_list=[0, 1, 2], method=Method.AUTO.value)
    assert explanation.method == Method.RECURSION


@pytest.mark.parametrize('predictor', [SVR()])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_targets, random_state',
    [(10, 100, 30, 1, 1)]
)
def test_method_auto_brute(predictor, n_samples, n_features, n_informative, n_targets, random_state):
    """ Checks if the method='auto' falls to method='brute' for some specific class of regressors. """
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_targets=n_targets,
                           random_state=random_state)
    predictor.fit(X, y)

    explainer = PartialDependence(predictor=predictor)
    explanation = explainer.explain(X=X, features_list=[0, 1, 2], method=Method.AUTO.value)
    assert explanation.method == Method.BRUTE


@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unsupported_method_recursion(predictor, iris_data):
    """ Checks if raises error when the method='recursion' for a classifier which does not support it. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X_train, features_list=[0, 1, 2], method=Method.RECURSION)


@pytest.mark.parametrize('predictor', [DecisionTreeRegressor(), RandomForestRegressor()])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_targets, random_state',
    [(10, 100, 30, 1, 1)]
)
def test_method_recursion_response_method_auto(predictor, n_samples, n_features, n_informative,
                                               n_targets, random_state):
    """ Checks if the response method='auto' falls to method='decision_function' for a classifier which
    supports method='recursion'. """
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_targets=n_targets,
                           random_state=random_state)
    predictor.fit(X, y)

    explainer = PartialDependence(predictor=predictor)
    explanation = explainer.explain(X=X,
                                    features_list=[0, 1, 2],
                                    method=Method.RECURSION,
                                    response_method=ResponseMethod.AUTO)
    assert explanation.response_method == ResponseMethod.DECISION_FUNCTION


@pytest.mark.parametrize('predictor', [DecisionTreeRegressor(), RandomForestRegressor()])
@pytest.mark.parametrize(
    'n_samples, n_features, n_informative, n_targets, random_state',
    [(10, 100, 30, 1, 1)]
)
def test_method_recursion_response_method_predict_proba(predictor, n_samples, n_features, n_informative,
                                                        n_targets, random_state):
    """ Checks if raises error when method='recursion' for a classifier which supports it and when
    the response_method='predict_proba'. """
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_targets=n_targets,
                           random_state=random_state)
    predictor.fit(X, y)

    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError):
        explainer.explain(X=X,
                          features_list=[0, 1, 2],
                          method=Method.RECURSION,
                          response_method=ResponseMethod.PREDICT_PROBA)
