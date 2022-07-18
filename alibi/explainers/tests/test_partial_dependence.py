import pytest
import re
import numpy as np
from pytest_lazyfixture import lazy_fixture

from alibi.explainers import PartialDependence
from alibi.explainers.partial_dependence import ResponseMethod, Kind, Method

from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputClassifier


@pytest.fixture(scope='module')
def multioutput_dataset():
    n_samples, n_features, n_informative, n_classes = 10, 100, 30, 3
    X, y1 = make_classification(n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                n_classes=n_classes,
                                random_state=0)

    y2 = shuffle(y1, random_state=1)
    y3 = shuffle(y1, random_state=2)
    Y = np.vstack((y1, y2, y3)).T
    return {'X': X,  'Y': Y}


@pytest.fixture(scope='module')
def multioutput_classifier(request):
    predictor = request.param
    return MultiOutputClassifier(predictor)


@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unfitted_estimator(predictor):
    """ Checks if raises error for unfitted model. """
    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(NotFittedError) as err:
        explainer._params_sanity_checks(estimator=predictor)
    assert re.search('not fitted yet', err.value.args[0])


@pytest.mark.parametrize('multioutput_classifier', [RandomForestClassifier()], indirect=True)
def test_multioutput_estimator(multioutput_classifier, multioutput_dataset):
    """ Check if raises error for multi-output model"""
    X, Y = multioutput_dataset['X'], multioutput_dataset['Y']
    multioutput_classifier.fit(X, Y)

    explainer = PartialDependence(predictor=multioutput_classifier)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=multioutput_classifier)
    assert re.search('multiclass-multioutput', err.value.args[0].lower())


@pytest.mark.parametrize('response_method', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_response_method(rf_classifier, response_method):
    """ Checks if raises error for unknown response_method. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor, response_method=response_method)
    assert re.search("response_method=\'\w+\' is invalid", err.value.args[0].lower())


@pytest.mark.parametrize('response_method', [ResponseMethod.DECISION_FUNCTION, ResponseMethod.PREDICT_PROBA])
@pytest.mark.parametrize('rf_regressor', [lazy_fixture('boston_data')], indirect=True)
def test_estimator_response_method(rf_regressor, response_method):
    """ Checks if raises error for a regressor with a response_method != 'auto'. """
    predictor, _ = rf_regressor
    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor, response_method=response_method)
    assert re.search('is ignored for regressor', err.value.args[0].lower())


@pytest.mark.parametrize('method', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_method(rf_classifier, method):
    """ Checks if raises error for unknown method. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor, method=method)
    assert re.search("method=\'\w+\' is invalid", err.value.args[0].lower())


@pytest.mark.parametrize('kind', [Kind.INDIVIDUAL, Kind.BOTH])
@pytest.mark.parametrize('method', [Method.RECURSION])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_kind_method(rf_classifier, kind, method):
    """ Checks if raises error when method='recursion' and kind !='average'. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor, kind=kind, method=method)
    assert re.search("when kind='average'", err.value.args[0].lower())


@pytest.mark.parametrize('predictor', [
    GradientBoostingRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
])
def test_method_auto_recursion(predictor, boston_data):
    """ Checks if the method='auto' falls to method='recursion' for some specific class of regressors. """
    X_train, y_train = boston_data['X_train'], boston_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    _, method, _ = explainer._params_sanity_checks(estimator=predictor, method=Method.AUTO.value)
    assert method == Method.RECURSION


@pytest.mark.parametrize('predictor', [SVR()])
def test_method_auto_brute(predictor, boston_data):
    """ Checks if the method='auto' falls to method='brute' for some specific class of regressors. """
    X_train, y_train = boston_data['X_train'], boston_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    _, method, _ = explainer._params_sanity_checks(estimator=predictor, method=Method.AUTO.value)
    assert method == Method.BRUTE


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unsupported_method_recursion(rf_classifier):
    """ Checks if raises error when the method='recursion' for a classifier which does not support it. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor, method=Method.RECURSION)
    assert re.search("support the 'recursion'", err.value.args[0].lower())


@pytest.mark.parametrize('predictor', [DecisionTreeRegressor(), RandomForestRegressor()])
def test_method_recursion_response_method_auto(predictor, boston_data):
    """ Checks if the response method='auto' falls to method='decision_function' for a classifier which
    supports method='recursion'. """
    X_train, y_train = boston_data['X_train'], boston_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    response_method, _, _ = explainer._params_sanity_checks(estimator=predictor,
                                                            method=Method.RECURSION,
                                                            response_method=ResponseMethod.AUTO)
    assert response_method == ResponseMethod.DECISION_FUNCTION


@pytest.mark.parametrize('predictor', [GradientBoostingClassifier()])
def test_method_recursion_response_method_predict_proba(predictor, iris_data):
    """ Checks if raises error when method='recursion' for a classifier which supports it and when
    the response_method='predict_proba'. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor)
    with pytest.raises(ValueError) as err:
        explainer._params_sanity_checks(estimator=predictor,
                                        method=Method.RECURSION,
                                        response_method=ResponseMethod.PREDICT_PROBA)
    assert re.search('the response_method must be', err.value.args[0].lower())
