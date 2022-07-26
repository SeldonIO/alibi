import re
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import pytest
from pytest_lazyfixture import lazy_fixture

from alibi.explainers import PartialDependence
from alibi.explainers.partial_dependence import ResponseMethod, Kind, Method, _sample_ice

from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture(scope='module')
def binary_data():
    n_samples, n_feautres, n_informative, n_classes = 200, 100, 30, 2
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_feautres,
                               n_informative=n_informative,
                               n_classes=n_classes,
                               random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': None,
    }


@pytest.fixture(scope='module')
def multioutput_dataset():
    n_samples, n_features, n_informative, n_classes = 200, 100, 30, 3
    X, y1 = make_classification(n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                n_classes=n_classes,
                                random_state=0)

    y2 = shuffle(y1, random_state=1)
    y3 = shuffle(y1, random_state=2)
    Y = np.vstack((y1, y2, y3)).T
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'preprocessor': None
    }


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
    X_train, Y_train = multioutput_dataset['X_train'], multioutput_dataset['Y_train']
    multioutput_classifier.fit(X_train, Y_train)

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
    assert re.search("response_method=\'\w+\' is invalid", err.value.args[0].lower())  # noqa: W605


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
    assert re.search("method=\'\w+\' is invalid", err.value.args[0].lower())  # noqa: W605


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


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [(0, 1, 2)],
    [0, (0, 1), (0, 1, 2)],
    [0, 1, 2, (0, 1, 2)],
    [0, 1, tuple()]
])
def test_num_features(rf_classifier, iris_data, features):
    """ Checks if raises error when a requested partial dependence for a tuple of containing more than two features or
    less than one. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor,
                                  feature_names=list(range(iris_data['X_train'].shape[1])))
    with pytest.raises(ValueError):
        explainer._features_sanity_checks(features=features)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('grid_resolution', [3, 5, np.inf])
@pytest.mark.parametrize('features', [
    [0, 1, (0, 1), (1, 2)]
])
def test_explanation_numerical_shapes(rf_classifier, iris_data, grid_resolution, features):
    """ Checks the correct shapes of the arrays contained in the explanation object for numerical features. """
    predictor, _ = rf_classifier
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    unique_labels = len(np.unique(y_train))
    num_targets = 1 if unique_labels == 2 else unique_labels
    num_instances = len(X_train)

    explanier = PartialDependence(predictor=predictor)
    exp = explanier.explain(X=X_train, features_list=features, grid_resolution=grid_resolution, kind=Kind.BOTH)

    # check that the values returned match the number of requested features
    assert len(exp.feature_names) == len(features)
    assert len(exp.pd_values) == len(features)
    assert len(exp.ice_values) == len(features)
    assert len(exp.feature_values) == len(features)

    for i, f in enumerate(features):
        if isinstance(f, Tuple):
            # check deciles
            assert isinstance(exp.feature_deciles[i], List)
            assert len(exp.feature_deciles[i]) == len(f)
            assert len(exp.feature_deciles[i][0]) == 11
            assert len(exp.feature_deciles[i][1]) == 11

            # check feature_values
            assert isinstance(exp.feature_values[i], List)
            assert len(exp.feature_values[i]) == len(f)
            assert len(exp.feature_values[i][0]) == (len(np.unique(X_train[:, f[0]]))
                                                     if grid_resolution == np.inf else grid_resolution)
            assert len(exp.feature_values[i][1]) == (len(np.unique(X_train[:, f[1]]))
                                                     if grid_resolution == np.inf else grid_resolution)

            # check pd_values
            assert exp.pd_values[i].shape == (num_targets,
                                              len(exp.feature_values[i][0]),
                                              len(exp.feature_values[i][1]))

            # check ice_values
            assert exp.ice_values[i].shape == (num_targets,
                                               num_instances,
                                               len(exp.feature_values[i][0]),
                                               len(exp.feature_values[i][1]))

        else:
            # check feature_deciles
            assert len(exp.feature_deciles[i]) == 11

            # check feature_values
            assert len(exp.feature_values[i]) == (len(np.unique(X_train[:, f]))
                                                  if grid_resolution == np.inf else grid_resolution)

            # check pd_values
            assert exp.pd_values[i].shape == (num_targets, len(exp.feature_values[i]))

            # check ice_value
            assert exp.ice_values[i].shape == (num_targets, num_instances, len(exp.feature_values[i]))


@pytest.mark.parametrize('rf_regressor', [lazy_fixture('boston_data')], indirect=True)
@pytest.mark.parametrize('kind', ['average', 'individual', 'both'])
@pytest.mark.parametrize('feature_list', [
    [0, 1, 2],
    [(0, 1), (0, 2), (1, 2)],
    [0, 1, (0, 1)]
])
def test_regression_wrapper(rf_regressor, boston_data, kind, feature_list):
    """ Test the black-box wrapper for a regression function. """
    rf, _ = rf_regressor
    predictor = rf.predict  # equivalent of black-box model
    X_train = boston_data['X_train']

    # define predictor kwargs
    predictor_kw = {
        'predictor_type': 'regressor',
        'prediction_fn': 'predict'
    }

    # define explainer and compute explanation
    explainer = PartialDependence(predictor=predictor, predictor_kw=predictor_kw)
    explainer.explain(X=X_train,
                      features_list=feature_list,
                      grid_resolution=10,
                      response_method='auto',
                      method='brute',
                      kind=kind)


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('response_method', ['auto', 'predict_proba', 'decision_function'])
@pytest.mark.parametrize('kind', ['average', 'individual', 'both'])
@pytest.mark.parametrize('feature_list', [
    [0, 1, 2],
    [(0, 1), (0, 2), (1, 2)],
    [0, 1, (0, 1)]
])
def test_classification_wrapper(lr_classifier, iris_data, response_method, kind, feature_list):
    """ Test the black-box wrapper for a classification function. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    lr, _ = lr_classifier
    num_classes = len(np.unique(y_train))

    if response_method == 'decision_function':
        predictor = lr.decision_function
        prediction_fn = 'decision_function'
    else:
        predictor = lr.predict_proba
        prediction_fn = 'predict_proba'

    # define predictor kwargs
    predictor_kw = {
        'predictor_type': 'classifier',
        'prediction_fn': prediction_fn,
        'num_classes': num_classes

    }

    # define explainer and compute explanation
    explainer = PartialDependence(predictor=predictor, predictor_kw=predictor_kw)
    explainer.explain(X=X_train,
                      features_list=feature_list,
                      grid_resolution=10,
                      response_method=response_method,
                      method='brute',
                      kind=kind)


@pytest.mark.parametrize('use_int', [False, True])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
def test_grid_points(adult_data, rf_classifier, use_int):
    """ Checks whether the grid points provided are used for computing the partial dependencies. """
    rf, _ = rf_classifier
    rf_clone = deepcopy(rf)  # need to deepcopy as the rf_classifier fixture has module scope

    def decorator(func):
        def wrapper(X, *args, **kwargs):
            X_ohe = adult_data['preprocessor'].transform(X)
            return func(X_ohe, *args, **kwargs)
        return wrapper

    # decorate predict_proba such that it accepts label encodings and transforms it internally to ohe
    rf_clone.predict_proba = decorator(rf_clone.predict_proba)

    feature_names = adult_data['metadata']['feature_names']
    categorical_names = adult_data['metadata']['category_map']
    X_train = adult_data['X_train']

    # construct random grid_points by choosing random values between min and max for each numerical feature,
    # and sampling at random from the categorical names for each categorical feature.
    grid_points = {}
    for i in range(len(feature_names)):
        if i not in categorical_names:
            min_val, max_val = X_train[:, i].min(), X_train[:, i].max()
            size = np.random.randint(low=1, high=len(np.unique(X_train[:, i])))
            vals = np.random.uniform(min_val, max_val, size=size)
        else:
            size = np.random.randint(low=1, high=len(categorical_names[i]))
            categorical_values = np.arange(len(categorical_names[i])) if use_int else categorical_names[i]
            vals = np.random.choice(categorical_values, size=size, replace=False)

        grid_points[i] = vals

    # define explainer
    explainer = PartialDependence(predictor=rf_clone,
                                  feature_names=feature_names,
                                  categorical_names=categorical_names)

    # compute explanation for every feature using the grid_points
    exp = explainer.explain(X=X_train[:100],
                            features_list=None,
                            response_method='predict_proba',
                            kind='average',
                            grid_points=grid_points)

    for i in range(len(feature_names)):
        np.testing.assert_allclose(exp.feature_values[i], grid_points[i])


@pytest.mark.parametrize('use_int', [False, True])
def test_grid_points_error(adult_data, use_int):
    """ Checks if the _grid_points_sanity_checks throw an error when the grid_points for a categorical feature
    are not a subset of the feature values provided in categorical_names. """
    feature_names = adult_data['metadata']['feature_names']
    categorical_names = adult_data['metadata']['category_map']
    X_train = adult_data['X_train']

    # construct random grid_points by choosing random values between min and max for each numerical feature,
    # and sampling at random from the categorical names for each categorical feature.
    grid_points = {}
    for i in range(len(feature_names)):
        if i in categorical_names:
            size = np.random.randint(low=1, high=len(categorical_names[i]))
            categorical_values = np.arange(len(categorical_names[i])) if use_int else categorical_names[i]
            grid_points[i] = np.random.choice(categorical_values, size=size, replace=False)

            # append a wrong value
            grid_points[i] = np.append(grid_points[i], len(categorical_values) if use_int else '[UNK]')

    # define explainer
    explainer = PartialDependence(predictor=lambda x: np.zeros(x.shape[0]),
                                  feature_names=feature_names,
                                  categorical_names=categorical_names,
                                  predictor_kw={
                                      'predictor_type': 'classifier',
                                      'prediction_fn': 'predict_proba',
                                      'num_classes': 2
                                  })

    # compute explanation for every feature using the grid_points
    with pytest.raises(ValueError):
        explainer._grid_points_sanity_checks(grid_points, n_features=X_train.shape[1])


@pytest.mark.parametrize('response_method', ['predict_proba', 'auto'])
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('binary_data')], indirect=True)
def test_binary_classifier_two_targets(lr_classifier, binary_data, response_method):
    """ Checks that for a classifier which has predict_proba and decision and for which the
    response_method in ['predict_proba', 'auto'], the partial dependence has two targets. """
    lr, _ = lr_classifier
    X_train = binary_data['X_train']

    explainer = PartialDependence(predictor=lr)
    exp = explainer.explain(X=X_train, response_method=response_method, kind='both')

    for pd, ice in zip(exp.pd_values, exp.ice_values):
        assert pd.shape[0] == 2
        assert ice.shape[0] == 2


@pytest.mark.parametrize('svc_classifier', [lazy_fixture('binary_data')], indirect=True)
def test_binary_classifier_one_target(svc_classifier, binary_data):
    """ Checks that for a classifier which does not have predict_proba and for which the response_method='auto',
    the partial dependence has only one target. """
    svc, _ = svc_classifier
    X_train = binary_data['X_train']

    explainer = PartialDependence(predictor=svc)
    exp = explainer.explain(X=X_train, response_method='auto', kind='both')

    for pd, ice in zip(exp.pd_values, exp.ice_values):
        assert pd.shape[0] == 1
        assert ice.shape[0] == 1


@pytest.mark.parametrize('n_ice', ['all', 'list', 'int'])
@pytest.mark.parametrize('n_samples, n_values', [(100, 10)])
def test_ice_sampling(n_ice, n_samples, n_values):
    """ Checks if the ice sampling helper function works properly when the arguments are valid. """
    ice_vals = np.random.randn(n_values, n_samples)

    if n_ice == 'all':
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice, seed=0)
        np.testing.assert_allclose(ice_vals, ice_sampled_vals)

    elif n_ice == 'list':
        size = np.random.randint(1, n_samples)
        # needs to be sorted because of the np.unique applied inside the _sample_ice
        n_ice = np.sort(np.random.choice(n_samples, size=size, replace=False)).tolist()
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice, seed=0)
        np.testing.assert_allclose(ice_vals[:, n_ice], ice_sampled_vals)

    else:
        n_ice = np.random.randint(1, n_samples)
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice, seed=0)
        assert ice_sampled_vals.shape[1] == n_ice


@pytest.mark.parametrize('n_ice', ['unknown', -10, [-1, 1, 2]])
@pytest.mark.parametrize('n_samples, n_values', [(100, 10)])
def test_ice_sampling_error(n_samples, n_values, n_ice):
    """ Checks if the ice sampling helper function throws an error when the arguments are invalid. """
    ice_vals = np.random.rand(n_values, n_samples)
    with pytest.raises(ValueError):
        _sample_ice(ice_values=ice_vals, n_ice=n_ice, seed=0)
